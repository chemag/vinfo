#!/usr/bin/env python3

"""vinfo.py: A script to dump information about a video file.

vinfo.py parses a video file, and dumps per-frame and per-second
information about it.

Implemented as a script around a few other CLI tools (ffprobe).
"""

import argparse
from collections import defaultdict
import numpy as np
import os
import re
import subprocess
import sys

FUNC_CHOICES = {
    'help': 'show help options',
    'frames': 'run frame analysis',
    'time': 'run frame analysis',
}


default_values = {
    'debug': 0,
    'dry_run': False,
    'stream_id': 'v:0',
    'period_frames': 30,
    'add_qp': True,
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

    if options.func == 'frames':
        # 1. get per-frame, qp information from ffprobe
        frame_list = get_frames_information(infile, options)
        if options.add_qp:
            qp_list = get_qp_information(infile, options)
            frame_list = join_frames_and_qp(frame_list, qp_list)
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
    command = 'ffprobe -select_streams %s -show_streams %s' % (
        options.stream_id, infile)
    returncode, out, err = run(command, options)
    assert returncode == 0, 'error running "%s"' % command
    # parse the output
    return parse_ffprobe_output(out, 'STREAM', options.debug)


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


# get QP information
def join_frames_and_qp(frame_list, qp_list):
    # join the lists (note that zip() stops at the smallest list)
    out = []
    for frame_info, qp_info in zip(frame_list, qp_list):
        # get the qp list as a numpy array
        qp_arr = np.array(qp_info[-1])
        frame_info['qp_min'] = qp_arr.min()
        frame_info['qp_max'] = qp_arr.max()
        frame_info['qp_mean'] = qp_arr.mean()
        frame_info['qp_var'] = qp_arr.var()
        out.append(frame_info)
    return out


def get_qp_information(infile, options):
    command = f'ffprobe -v quiet -show_frames -debug qp {infile}'
    returncode, out, err = run(command, options)
    if returncode != 0:
        raise InvalidCommand('error running "%s"' % command)
    # parse the output
    return parse_qp_information(err, options.debug)


def parse_qp_information(out, debug):
    qp_full = []
    cur_frame = -1
    resolution = None
    pix_fmt = None
    frame_type = None
    qp_vals = []

    reinit_pattern = (
        r'\[[^\]]+\] Reinit context to (?P<resolution>\d+x\d+), '
        r'pix_fmt: (?P<pix_fmt>.+)'
    )
    newframe_pattern = (
        r'\[[^\]]+\] New frame, type: (?P<frame_type>.+)'
    )
    qp_pattern = (
        r'\[[^\]]+\] (?P<qp_str>\d+)'
    )

    for line in out.splitlines():
        line = line.decode('ascii').strip()
        if 'Reinit context to' in line:
            # [h264 @ 0x30d1a80] Reinit context to 1280x720, pix_fmt: yuv420p
            match = re.search(reinit_pattern, line)
            if not match:
                print('warning: invalid reinit line ("%s")' % line)
                sys.exit(-1)
            resolution = match.group('resolution')
            pix_fmt = match.group('pix_fmt')

        elif 'New frame, type:' in line:
            # [h264 @ 0x30d1a80] New frame, type: I
            match = re.search(newframe_pattern, line)
            if not match:
                print('warning: invalid newframe line ("%s")' % line)
                sys.exit(-1)
            # store the old frame info
            if cur_frame != -1:
                qp_full.append([cur_frame, resolution, pix_fmt, frame_type,
                                qp_vals])
                qp_vals = []
            # new frame
            frame_type = match.group('frame_type')
            cur_frame += 1

        else:
            # [h264 @ 0x30d1a80] 3535353535353535353535...
            match = re.search(qp_pattern, line)
            if not match:
                continue
            qp_str = match.group('qp_str')
            qp_vals += [int(qp_str[i:i+2]) for i in range(0, len(qp_str), 2)]

    # dump the last state
    if qp_vals:
        qp_full.append([cur_frame, resolution, pix_fmt, frame_type, qp_vals])
        qp_vals = []

    return qp_full


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
