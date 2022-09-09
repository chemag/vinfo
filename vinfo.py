#!/usr/bin/env python3

"""vinfo.py: A script to dump information about a video file.

vinfo.py parses a video file, and dumps per-frame and per-second
information about it.

Implemented as a script around a few other CLI tools (ffprobe).
"""

import argparse
from collections import defaultdict
import numpy as np
import math
import os
import re
import subprocess
import sys

FUNC_CHOICES = {
    'help': 'show help options',
    'streams': 'run stream analysis',
    'frames': 'run frame analysis',
    'time': 'run frame analysis',
}


default_values = {
    'debug': 0,
    'dry_run': False,
    'stream_id': 'v:0',
    'period_frames': 30,
    'add_qp': True,
    'add_bpp': True,
    'add_motion_vec': True,
    'add_mb_type': True,
    'func': 'help',
    'infile': None,
    'outfile': None,
}


def run(command, options, **kwargs):
    env = kwargs.get('env', None)
    stdin = subprocess.PIPE if kwargs.get('stdin', False) else None
    bufsize = kwargs.get('bufsize', 0)
    universal_newlines = kwargs.get('universal_newlines', False)
    default_close_fds = True if sys.platform == 'linux2' else False
    close_fds = kwargs.get('close_fds', default_close_fds)
    shell = type(command) in (type(''), type(u''))
    if options.debug > 0:
        print('running $ %s' % command)
    if options.dry_run:
        return 0, b'stdout', b'stderr'
    p = subprocess.Popen(command, stdin=stdin,  # noqa: P204
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, bufsize=bufsize,
                         universal_newlines=universal_newlines,
                         env=env, close_fds=close_fds, shell=shell)
    # wait for the command to terminate
    if stdin is not None:
        out, err = p.communicate(stdin)
    else:
        out, err = p.communicate()
    returncode = p.returncode
    # clean up
    del p
    # return results
    return returncode, out, err


class InvalidCommand(Exception):
    pass


def parse_file(infile, outfile, options):
    # 0. make sure file exists
    assert os.access(infile, os.R_OK), 'error: file %s does not exist' % infile

    if options.func == 'streams':
        # 1. get per-stream information from ffprobe
        stream_list = get_streams_information(infile, options)
        # 2. dump all information together as streams
        with open(outfile, 'w') as f:
            # get all the possible keys
            key_list = list(stream_list[0].keys())
            for stream_info in stream_list:
                for key in stream_info:
                    if key not in key_list:
                        key_list.append(key)
            # write the header
            header_format = '# %s\n' % ','.join(['%s'] * len(key_list))
            f.write(header_format % tuple(key_list))
            # write the line format
            line_format = '{' + '},{'.join(key_list) + '}\n'
            # write all the lines
            for stream_info in stream_list:
                d = defaultdict(str, **stream_info)
                f.write(line_format.format_map(d))

    if options.func == 'frames':
        # 1. get per-frame, qp information from ffprobe
        frame_list = get_frames_information(infile, options)
        if options.add_bpp:
            frame_list = add_bpp_column(frame_list)
        if options.add_qp:
            qp_list = get_qp_information(infile, options)
            frame_list = join_frames_and_qp(frame_list, qp_list)
        if options.add_mb_type:
            mb_list = get_mb_information(infile, options)
            frame_list = join_frames_and_mb(frame_list, mb_list)
        if options.add_motion_vec:
            mv_list = get_mv_information(infile, options)
            frame_list = join_frames_and_mv(frame_list, mv_list)
        # 2. dump all information together as frames
        with open(outfile, 'w') as f:
            # get all the possible keys
            key_list = list(frame_list[0].keys())
            for frame_info in frame_list:
                for key in frame_info:
                    if key not in key_list:
                        key_list.append(key)
            # write the header
            header_format = '# %s\n' % ','.join(['%s'] * len(key_list))
            f.write(header_format % tuple(key_list))
            # write the line format
            line_format = '{' + '},{'.join(key_list) + '}\n'
            # write all the lines
            for frame_info in frame_list:
                d = defaultdict(str, **frame_info)
                f.write(line_format.format_map(d))

    if options.func == 'time':
        # 1. get per-frame information from ffprobe
        frame_list = get_frames_information(infile, options)
        # 2. dump all information aggregated by time
        time_frame_list = aggregate_list_by_frame_number(
            frame_list, 'frame_number', options.period_frames)
        with open(outfile, 'w') as f:
            # aggregated values
            time_key_list = list(time_frame_list[0].keys())
            header_format = '# %s\n' % ','.join(['%s'] * len(time_key_list))
            f.write(header_format % tuple(time_key_list))
            line_format = ','.join(['%s'] * len(time_key_list)) + '\n'
            for time_frame_info in time_frame_list:
                f.write(line_format % tuple(time_frame_info.values()))


def aggregate_list_by_frame_number(in_list, field, period):
    # 1. calculate output data
    frame_list = []
    last_agg_value = None
    cum_frames = 0
    cum_bytes = 0
    for in_info in in_list:
        if last_agg_value is None:
            last_agg_value = int(in_info[field])
        if int(in_info[field]) >= (last_agg_value + period):
            # dump value
            frame_info = {
                field: last_agg_value,
                'num_frames': cum_frames,
                'pkt_size': cum_bytes,
            }
            frame_list.append(frame_info)
            cum_frames = 0
            cum_bytes = 0
            # insert zeroes where no data is present
            delta_number = int(in_info[field]) - last_agg_value
            zero_elements = int((delta_number - period) / period)
            for _i in range(zero_elements):
                last_agg_value += period
                frame_info = {
                    field: last_agg_value,
                    'num_frames': 0,
                    'pkt_size': 0,
                }
                frame_list.append(frame_info)
            last_agg_value += period
        # account for current packet
        cum_frames += 1
        cum_bytes += int(in_info['pkt_size'])
    # flush data
    if cum_frames > 0:
        frame_info = {
            field: last_agg_value,
            'num_frames': cum_frames,
            'pkt_size': cum_bytes,
        }
        frame_list.append(frame_info)
    return frame_list


# get video information
def get_streams_information(infile, options):
    command = 'ffprobe -select_streams %s -count_frames -show_streams %s' % (
        options.stream_id, infile)
    returncode, out, err = run(command, options)
    assert returncode == 0, 'error running "%s"' % command
    # parse the output
    stream_list = parse_ffprobe_output(out, 'STREAM', options.debug)
    for stream_info in stream_list:
        # 1. remove useless keys
        rem_key_list = list(stream_info.keys())
        rem_key_list = [key for key in rem_key_list if
                        key.startswith('DISPOSITION:')]
        for key in rem_key_list:
            del stream_info[key]
        # 2. add interesting keys
        resolution_pixels_per_frame = (int(stream_info['width']) *
                                       int(stream_info['height']))
        file_size_bytes = os.path.getsize(infile)
        file_size_bits = 8 * file_size_bytes
        num_frames = int(stream_info['nb_read_frames'])
        bits_per_pixel = file_size_bits / (
            resolution_pixels_per_frame * num_frames)
        frames_per_second = 30
        bitrate_bps = file_size_bits * frames_per_second / num_frames
        stream_info['resolution'] = resolution_pixels_per_frame
        stream_info['filesize'] = file_size_bytes
        stream_info['bpp'] = bits_per_pixel
        stream_info['bitrate_mbps'] = bitrate_bps / 1e6

    return stream_list


def get_frames_information(infile, options):
    command = 'ffprobe -select_streams %s -show_frames %s' % (
        options.stream_id, infile)
    returncode, out, err = run(command, options)
    assert returncode == 0, 'error running "%s"' % command
    # parse the output
    return parse_ffprobe_per_frame_info(out, options.debug)


def parse_ffprobe_per_frame_info(out, debug):
    frame_list = parse_ffprobe_output(out, 'FRAME', debug)
    # add frame numbers
    frame_number = 0
    new_frame_list = []
    for frame_info in frame_list:
        new_frame_info = {
            'frame_number': frame_number,
        }
        new_frame_info.update(frame_info)
        # add bits per pixel (bpp)
        new_frame_info['bpp'] = ((int(new_frame_info['pkt_size']) * 8) /
                                 (int(new_frame_info['width']) *
                                  int(new_frame_info['height'])))
        new_frame_list.append(new_frame_info)
        frame_number += 1
    return new_frame_list


def parse_ffprobe_output(out, label, debug):
    item_list = []
    item_info = {}
    start_item = '[%s]' % label
    end_item = '[/%s]' % label
    for line in out.splitlines():
        line = line.decode('ascii').strip()
        if line == start_item:
            item_info = {}
        elif line == end_item:
            item_list.append(item_info)
        elif '=' in line:
            key, value = line.split('=', 1)
            item_info[key] = value
        else:
            if debug > 0:
                print('warning: unknown line ("%s")' % line)
    return item_list


# BPP information
def add_bpp_column(frame_list):
    out = []
    for frame_info in frame_list:
        width = int(frame_info['width'])
        height = int(frame_info['height'])
        pkt_size = int(frame_info['pkt_size'])
        # bytes/frame * bits/byte / pixels/frame = bits/pixel
        bpp = (pkt_size * 8) / (width * height)
        frame_info['bpp'] = bpp
        out.append(frame_info)
    return out


# get QP information
def join_frames_and_qp(frame_list, qp_list):
    # ensure both lists have the same length
    assert len(frame_list) == len(qp_list), (
        f'join error: different sizes {len(frame_list)} != {len(qp_list)}')
    # join the lists (note that zip() stops at the smallest list)
    common_key_list = set(frame_list[0].keys()) & set(qp_list[0].keys())
    out = []
    for frame_info, qp_info in zip(frame_list, qp_list):
        # ensure all the common keys have the same values
        for key in common_key_list:
            assert frame_info[key] == qp_info[key], (
                'join error: different value '
                f'frame_info["{key}"] = {frame_info[key]} != '
                f'qp_info["{key}"] = {qp_info[key]}')
        # get the qp list as a numpy array
        qp_arr = np.array(qp_info['qp_vals'])
        frame_info['qp_min'] = qp_arr.min()
        frame_info['qp_max'] = qp_arr.max()
        frame_info['qp_mean'] = qp_arr.mean()
        frame_info['qp_var'] = qp_arr.var()
        out.append(frame_info)
    return out


def join_frames_and_mb(frame_list, mb_list):
    # ensure both lists have the same length
    assert len(frame_list) == len(mb_list), (
        f'join error: different sizes {len(frame_list)} != {len(mb_list)}')
    # join the lists (note that zip() stops at the smallest list)
    common_key_list = set(frame_list[0].keys()) & set(mb_list[0].keys())
    out = []
    for frame_info, mb_info in zip(frame_list, mb_list):
        # ensure all the common keys have the same values
        for key in common_key_list:
            assert frame_info[key] == mb_info[key], (
                'join error: different value '
                f'frame_info["{key}"] = {frame_info[key]} != '
                f'mb_info["{key}"] = {mb_info[key]}')
        frame_info.update(mb_info['mb_info'])
        out.append(frame_info)
    return out


def join_frames_and_mv(frame_list, mv_list):
    # ensure both lists have the same length
    assert len(frame_list) == len(mv_list), (
        f'join error: different sizes {len(frame_list)} != {len(mv_list)}')
    # join the lists (note that zip() stops at the smallest list)
    out = []
    common_key_list = set(frame_list[0].keys()) & set(mv_list[0].keys())
    for frame_info, mv_info in zip(frame_list, mv_list):
        # ensure all the common keys have the same values
        for key in common_key_list:
            assert frame_info[key] == mv_info[key], (
                'join error: different value '
                f'frame_info["{key}"] = {frame_info[key]} != '
                f'mv_info["{key}"] = {mv_info[key]}')
        # get the mv list as a numpy array
        mvx_arr = np.array(mv_info['mv_vals_x'])
        mvy_arr = np.array(mv_info['mv_vals_y'])
        # TODO(chemag): stack arrays into (<len>, 2) instead of (1, <len>, 2)
        mvxy_arr = np.dstack((mvx_arr, mvy_arr))
        # TODO(chemag): there has to be a better numpy way to do this
        frame_mb_total = len(mvxy_arr[0])
        frame_mb_nonzero = 0
        frame_mod_sum = 0
        frame_max_x = 0
        frame_max_y = 0
        for (x, y) in mvxy_arr[0]:
            # common case: no motion vector
            if x == 0 and y == 0:
                continue
            frame_mb_nonzero += 1
            mod = abs(y if x == 0 else (x if y == 0 else
                                        math.sqrt(x ** 2 + y ** 2)))
            # print(f'x: {x}  y: {y}  mod: {mod}')
            frame_max_x = max(frame_max_x, abs(x))
            frame_max_y = max(frame_max_y, abs(y))
            frame_mod_sum += mod
        frame_mod_avg = (frame_mod_sum / frame_mb_nonzero if
                         frame_mb_nonzero else 0.0)
        # frame_mb_zero = frame_mb_total - frame_mb_nonzero
        frame_info['mv_nonzero_ratio'] = frame_mb_nonzero / frame_mb_total
        frame_info['mv_mod_average'] = frame_mod_avg
        frame_info['mv_max_x'] = frame_max_x
        frame_info['mv_max_y'] = frame_max_y
        out.append(frame_info)
    return out


def get_qp_information(infile, options):
    command = f'ffprobe -v quiet -show_frames -debug qp {infile}'
    returncode, out, err = run(command, options)
    if returncode != 0:
        raise InvalidCommand('error running "%s"' % command)
    # parse the output
    return parse_qp_information(err, options.debug)


def get_mb_information(infile, options):
    command = f'ffprobe -v quiet -show_frames -debug mb_type {infile}'
    returncode, out, err = run(command, options)
    if returncode != 0:
        raise InvalidCommand('error running "%s"' % command)
    # parse the output
    return parse_mb_information(err, options.debug)


def get_mv_information(infile, options):
    command = f'ffprobe -v quiet -show_frames -debug motion_vec {infile}'
    returncode, out, err = run(command, options)
    if returncode != 0:
        raise InvalidCommand('error running "%s"' % command)
    # parse the output
    return parse_mv_information(err, options.debug)


def parse_qp_information(out, debug):
    qp_full = []
    frame_number = -1
    resolution = None
    pix_fmt = None
    pict_type = None
    qp_vals = []

    reinit_pattern = (
        r'\[[^\]]+\] Reinit context to (?P<resolution>\d+x\d+), '
        r'pix_fmt: (?P<pix_fmt>.+)'
    )
    newframe_pattern = (
        r'\[[^\]]+\] New frame, type: (?P<pict_type>.+)'
    )
    qp_pattern = (
        r'\[[^\]]+\] (?P<qp_str>\d+)$'
    )

    for line in out.splitlines():
        line = line.decode('ascii').strip()
        if 'Reinit context to' in line:
            # [h264 @ 0x30d1a80] Reinit context to 1280x720, pix_fmt: yuv420p
            match = re.search(reinit_pattern, line)
            if not match:
                print('warning: invalid reinit line ("%s")' % line)
                sys.exit(-1)
            # reinit: flush all previous data
            resolution = match.group('resolution')
            pix_fmt = match.group('pix_fmt')
            qp_full = []
            frame_number = -1
            qp_vals = []

        elif 'New frame, type:' in line:
            # [h264 @ 0x30d1a80] New frame, type: I
            match = re.search(newframe_pattern, line)
            if not match:
                print('warning: invalid newframe line ("%s")' % line)
                sys.exit(-1)
            # store the old frame info
            if frame_number != -1:
                qp_full.append({
                    'frame_number': frame_number,
                    # TODO(chemag): resolution here does not consider cropping
                    # 'width': resolution.split('x')[0],
                    # 'height': resolution.split('x')[1],
                    'pix_fmt': pix_fmt,
                    'pict_type': pict_type,
                    'qp_vals': qp_vals})
                qp_vals = []
            # new frame
            pict_type = match.group('pict_type')
            frame_number += 1

        else:
            # [h264 @ 0x30d1a80] 3535353535353535353535...
            match = re.search(qp_pattern, line)
            if not match:
                continue
            qp_str = match.group('qp_str')
            qp_vals += [int(qp_str[i:i+2]) for i in range(0, len(qp_str), 2)]

    # dump the last state
    if qp_vals:
        qp_full.append({
            'frame_number': frame_number,
            # TODO(chemag): resolution here does not consider cropping
            # 'width': resolution.split('x')[0],
            # 'height': resolution.split('x')[1],
            'pix_fmt': pix_fmt,
            'pict_type': pict_type,
            'qp_vals': qp_vals})
        qp_vals = []

    return qp_full


MB_TYPE_LIST = [
  'P',  # IS_PCM(mb_type)  // MB_TYPE_INTRA_PCM
  'A',  # IS_INTRA(mb_type) && IS_ACPRED(mb_type)  // MB_TYPE_ACPRED
  'i',  # IS_INTRA4x4(mb_type)  // MB_TYPE_INTRA4x4
  'I',  # IS_INTRA16x16(mb_type)  // MB_TYPE_INTRA16x16
  'd',  # IS_DIRECT(mb_type) && IS_SKIP(mb_type)
  'D',  # IS_DIRECT(mb_type)  // MB_TYPE_DIRECT2
  'g',  # IS_GMC(mb_type) && IS_SKIP(mb_type)
  'G',  # IS_GMC(mb_type)  // MB_TYPE_GMC
  'S',  # IS_SKIP(mb_type)  // MB_TYPE_SKIP
  '>',  # !USES_LIST(mb_type, 1)
  '<',  # !USES_LIST(mb_type, 0)
  'X',  # av_assert2(USES_LIST(mb_type, 0) && USES_LIST(mb_type, 1))
]

MB_TYPE_SIMPLIFIED_DICT = {
    'intra': ['A', 'i', 'I', ],
    'intra-pcm': ['P', ],
    'inter': ['<', '>', ],
    'skip-direct': ['S', 'd', 'D', ],
    'other': ['X', ],
    'gmc': ['g', 'G', ],
}


def parse_mb_information(out, debug):
    mb_full = []
    frame_number = -1
    resolution = None
    pix_fmt = None
    pict_type = None
    mb_dict = {}

    reinit_pattern = (
        r'\[[^\]]+\] Reinit context to (?P<resolution>\d+x\d+), '
        r'pix_fmt: (?P<pix_fmt>.+)'
    )
    newframe_pattern = (
        r'\[[^\]]+\] New frame, type: (?P<pict_type>.+)'
    )
    mb_pattern = (
        r'\[[^\]]+\] (?P<mb_str>[PAiIdDgGS><X+\-|= ]+)$'
    )

    for line in out.splitlines():
        line = line.decode('ascii').strip()
        if 'Reinit context to' in line:
            # [h264 @ 0x30d1a80] Reinit context to 1280x720, pix_fmt: yuv420p
            match = re.search(reinit_pattern, line)
            if not match:
                print('warning: invalid reinit line ("%s")' % line)
                sys.exit(-1)
            # reinit: flush all previous data
            resolution = match.group('resolution')
            pix_fmt = match.group('pix_fmt')
            mb_full = []
            frame_number = -1
            mb_dict = {}

        elif 'New frame, type:' in line:
            # [h264 @ 0x30d1a80] New frame, type: I
            match = re.search(newframe_pattern, line)
            if not match:
                print('warning: invalid newframe line ("%s")' % line)
                sys.exit(-1)
            # store the old frame info
            if frame_number != -1:
                mb_info = {}
                for mb_type in MB_TYPE_LIST:
                    mb_info[f'mb_type_{mb_type}'] = (mb_dict.get(mb_type, 0) /
                                                     sum(mb_dict.values()))
                # calculate the derived values
                for mb_type2 in MB_TYPE_SIMPLIFIED_DICT.keys():
                    mb_info[f'mb_type_{mb_type2}'] = 0
                for mb_type2, mb_type_list in MB_TYPE_SIMPLIFIED_DICT.items():
                    for mb_type in mb_type_list:
                        mb_info[f'mb_type_{mb_type2}'] += (
                            mb_info[f'mb_type_{mb_type}'])
                mb_full.append({
                    'frame_number': frame_number,
                    # TODO(chemag): resolution here does not consider cropping
                    # 'width': resolution.split('x')[0],
                    # 'height': resolution.split('x')[1],
                    'pix_fmt': pix_fmt,
                    'pict_type': pict_type,
                    'mb_info': {**mb_info}})
                mb_dict = {}
            # new frame
            pict_type = match.group('pict_type')
            frame_number += 1

        else:
            # "[h264 @ ...] S  S  S  S  S  >- S  S  S  S  S  S  >  S  S  S  "
            match = re.search(mb_pattern, line)
            if not match:
                # print(f'error: invalid line: {line}')
                continue
            mb_str = match.group('mb_str')
            # make sure mb_str length is a multiple of 3
            while (len(mb_str) % 3) != 0:
                mb_str += ' '
            mb_list = [mb_str[i:i+1] for i in range(0, len(mb_str), 3)]
            row_mb_dict = {mb_type: mb_list.count(mb_type) for mb_type in
                           mb_list}
            for k, v in row_mb_dict.items():
                if k not in mb_dict:
                    mb_dict[k] = 0
                mb_dict[k] += v

    # dump the last state
    if mb_dict:
        mb_info = {}
        for mb_type in MB_TYPE_LIST:
            mb_info[f'mb_type_{mb_type}'] = (mb_dict.get(mb_type, 0) /
                                             sum(mb_dict.values()))
        # calculate the derived values
        for mb_type2 in MB_TYPE_SIMPLIFIED_DICT.keys():
            mb_info[f'mb_type_{mb_type2}'] = 0
        for mb_type2, mb_type_list in MB_TYPE_SIMPLIFIED_DICT.items():
            for mb_type in mb_type_list:
                mb_info[f'mb_type_{mb_type2}'] += (
                    mb_info[f'mb_type_{mb_type}'])
        mb_full.append({
            'frame_number': frame_number,
            # TODO(chemag): resolution here does not consider cropping
            # 'width': resolution.split('x')[0],
            # 'height': resolution.split('x')[1],
            'pix_fmt': pix_fmt,
            'pict_type': pict_type,
            'mb_info': {**mb_info}})
        mb_dict = {}

    return mb_full


def parse_mv_information(out, debug):
    mv_full = []
    frame_number = -1
    resolution = None
    pix_fmt = None
    pict_type = None
    mv_vals_x = []
    mv_vals_y = []

    reinit_pattern = (
        r'\[[^\]]+\] Reinit context to (?P<resolution>\d+x\d+), '
        r'pix_fmt: (?P<pix_fmt>.+)'
    )
    newframe_pattern = (
        r'\[[^\]]+\] New frame, type: (?P<pict_type>.+)'
    )
    mv_pattern = (
        r'\[[^\]]+\] (?P<mv_str>[\d\- ]+)$'
    )

    for line in out.splitlines():
        line = line.decode('ascii').strip()
        if 'Reinit context to' in line:
            # [h264 @ 0x30d1a80] Reinit context to 1280x720, pix_fmt: yuv420p
            match = re.search(reinit_pattern, line)
            if not match:
                print('warning: invalid reinit line ("%s")' % line)
                sys.exit(-1)
            # reinit: flush all previous data
            resolution = match.group('resolution')
            pix_fmt = match.group('pix_fmt')
            mv_full = []
            frame_number = -1
            mv_vals_x = []
            mv_vals_y = []

        elif 'New frame, type:' in line:
            # [h264 @ 0x30d1a80] New frame, type: I
            match = re.search(newframe_pattern, line)
            if not match:
                print('warning: invalid newframe line ("%s")' % line)
                sys.exit(-1)
            # store the old frame info
            if frame_number != -1:
                mv_full.append({
                    'frame_number': frame_number,
                    # TODO(chemag): resolution here does not consider cropping
                    # 'width': resolution.split('x')[0],
                    # 'height': resolution.split('x')[1],
                    'pix_fmt': pix_fmt,
                    'pict_type': pict_type,
                    'mv_vals_x': mv_vals_x,
                    'mv_vals_y': mv_vals_y})
                mv_vals_x = []
                mv_vals_y = []
            # new frame
            pict_type = match.group('pict_type')
            frame_number += 1

        else:
            # [h264 @ 0x30d1a80]     0    0   0    0 -12   -2   0    0
            match = re.search(mv_pattern, line)
            if not match:
                continue
            mv_str = match.group('mv_str')
            mv_list = []
            for s in mv_str.split():
                while '-' in s[1:]:
                    # process cases where the minus sign happens after a digit
                    index = s.index('-', 1)
                    mv_list.append(int(s[:index]))
                    s = s[index:]
                mv_list.append(int(s))
            mv_vals_x += mv_list[::2]
            mv_vals_y += mv_list[1::2]

    # dump the last state
    if mv_vals_x:
        mv_full.append({
            'frame_number': frame_number,
            # TODO(chemag): resolution here does not consider cropping
            # 'width': resolution.split('x')[0],
            # 'height': resolution.split('x')[1],
            'pix_fmt': pix_fmt,
            'pict_type': pict_type,
            'mv_vals_x': mv_vals_x,
            'mv_vals_y': mv_vals_y})
        mv_vals_x = []
        mv_vals_y = []

    return mv_full


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--debug', action='count',
        dest='debug', default=default_values['debug'],
        help='Increase verbosity (multiple times for more)',)
    parser.add_argument(
        '--quiet', action='store_const',
        dest='debug', const=-1,
        help='Zero verbosity',)
    parser.add_argument(
        '-D', '--dry-run', action='store_true',
        dest='dry_run', default=default_values['dry_run'],
        help='Dry run',)
    parser.add_argument(
        '--stream-id', action='store', type=str,
        dest='stream_id',
        default=default_values['stream_id'],
        metavar='STREAM_ID',
        help='stream ID',)
    parser.add_argument(
        '--period-frames', action='store', type=int,
        dest='period_frames',
        default=default_values['period_frames'],
        metavar='PERIOD_FRAMES',
        help='period in frames',)
    parser.add_argument(
        '--add-qp', action='store_const', default=default_values['add_qp'],
        dest='add_qp', const=True,
        help='Add QP columns (min, max, mean, var)',)
    parser.add_argument(
        '--noadd-qp', action='store_const',
        dest='add_qp', const=False,
        help='Do not add QP columns (min, max, mean, var)',)
    parser.add_argument(
        '--add-bpp', action='store_const', default=default_values['add_bpp'],
        dest='add_bpp', const=True,
        help='Add BPP column (bits per pixel)',)
    parser.add_argument(
        '--noadd-bpp', action='store_const',
        dest='add_bpp', const=False,
        help='Do not add BPP column (bits per pixel)',)
    parser.add_argument(
        '--add-mb-type', action='store_const',
        default=default_values['add_mb_type'],
        dest='add_mb_type', const=True,
        help='Add MB type columns',)
    parser.add_argument(
        '--noadd-mb-type', action='store_const',
        dest='add_mb_type', const=False,
        help='Do not add MB type columns',)
    parser.add_argument(
        '--add-motion-vec', action='store_const',
        default=default_values['add_motion_vec'],
        dest='add_motion_vec', const=True,
        help='Add motion vector columns',)
    parser.add_argument(
        '--noadd-motion-vec', action='store_const',
        dest='add_motion_vec', const=False,
        help='Do not add motion vector columns',)
    parser.add_argument(
        'func', type=str,
        default=default_values['func'],
        choices=FUNC_CHOICES.keys(),
        help='%s' % (' | '.join("{}: {}".format(k, v) for k, v in
                                FUNC_CHOICES.items())),)
    parser.add_argument(
        'infile', type=str, nargs='?',
        default=default_values['infile'],
        metavar='input-file',
        help='input file',)
    parser.add_argument(
        'outfile', type=str, nargs='?',
        default=default_values['outfile'],
        metavar='output-file',
        help='output file',)

    # do the parsing
    options = parser.parse_args(argv[1:])
    # implement help
    if options.func == 'help':
        parser.print_help()
        sys.exit(0)
    return options


def main(argv):
    # parse options
    options = get_options(argv)

    # get infile/outfile
    if options.infile == '-':
        options.infile = '/dev/fd/0'
    if options.outfile == '-':
        options.outfile = '/dev/fd/1'
    # print results
    if options.debug > 0:
        print(options)

    # do something
    parse_file(options.infile, options.outfile, options)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
