#!/usr/bin/env python3

"""vinfo.py: A script to dump information about a video file.

vinfo.py parses a video file, and dumps per-frame and per-second
information about it.

Implemented as a script around a few other CLI tools (ffprobe,
qpextract, etc. ).

"""

import argparse
from collections import defaultdict
import os
import re
import subprocess
import sys

FUNC_CHOICES = {
    'help': 'show help options',
    'frames': 'run frame analysis',
    'time': 'run frame analysis',
    'qp': 'run qp analysis',
}


default_values = {
    'debug': 0,
    'dry_run': False,
    'stream_id': 'v:0',
    'period_frames': 30,
    'func': 'help',
    'infile': None,
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


def parse_file(infile, options):
    # 0. make sure file exists
    assert os.access(infile, os.R_OK), 'error: file %s does not exist' % infile
    # 1. get stream information from ffprobe
    streams_list = get_streams_information(infile, options)

    # 2. get per-frame information from ffprobe
    frame_list = get_frames_information(infile, options)

    # 3. get QP information from qpextract (hevc-only so far)
    if streams_list[0]['codec_name'] == 'hevc':
        try:
            qp_list = get_qpextract_information(infile, options)
            if not qp_list:
                # qpextract is very picky (only accepts .265 files)
                0
            else:
                # zip information together
                frame_list = [{**ffprobe_info, **qp_info}
                              for ffprobe_info, qp_info in
                              zip(frame_list, qp_list)]
        except InvalidCommand:
            print('warning: could not run qpextract in %s' % infile)
            pass

    if options.func == 'frames':
        # 4. dump all information together as frames
        output_file = '%s.%s.csv' % (options.infile, 'frames')
        with open(output_file, 'w') as f:
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
        # 4. dump all information aggregated by time
        time_frame_list = aggregate_list_by_frame_number(
            frame_list, 'frame_number', options.period_frames)
        output_file = '%s.%s.csv' % (options.infile, 'time')
        with open(output_file, 'w') as f:
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
def get_qpextract_information(infile, options):
    command = 'qpextract %s' % infile
    returncode, out, err = run(command, options)
    if returncode != 0:
        raise InvalidCommand('error running "%s"' % command)
    # parse the output
    return parse_qpextract_per_frame_info(out, options)


def get_qpavg(qpmin, qpmax, qphisto):
    total_qp = 0
    total_values = 0
    qp = int(qpmin)
    for qp_num in qphisto:
        total_qp += (int(qp_num) * qp)
        total_values += int(qp_num)
        qp += 1
    return 1.0 * total_qp / total_values


def parse_qpextract_per_frame_info(out, options):
    frame_list = []
    # example: 'id: 0 qp_distro[26:35] { 11 21 25 89 73 70 243 136 170 3171 }'
    qpextract_pattern = (
        r'id: (?P<frame_number>\d+) '
        r'qp_distro\[(?P<qpmin>\d+):(?P<qpmax>\d+)\] '
        r'{ (?P<qphisto>[\d ]+) }'
    )
    for line in out.splitlines():
        line = line.decode('ascii').strip()
        match = re.search(qpextract_pattern, line)
        if not match:
            if options.debug > 0:
                print('warning: invalid line ("%s")' % line)
            continue
        frame_info = {
            'frame_number': match.group('frame_number'),
            'qpmin': match.group('qpmin'),
            'qpmax': match.group('qpmax'),
            'qpavg': get_qpavg(match.group('qpmin'), match.group('qpmax'),
                               match.group('qphisto').split(' ')),
            'qphisto': ':'.join(match.group('qphisto').split(' ')),
        }
        frame_list.append(frame_info)
    return frame_list


def get_options(argv):
    """Generic option parser.

    Args:
        argv: list containing arguments

    Returns:
        Namespace - An argparse.ArgumentParser-generated option object
    """
    # init parser
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--debug', action='count',
                        dest='debug', default=default_values['debug'],
                        help='Increase verbosity (multiple times for more)',)
    parser.add_argument('--quiet', action='store_const',
                        dest='debug', const=-1,
                        help='Zero verbosity',)
    parser.add_argument('-D', '--dry-run', action='store_true',
                        dest='dry_run', default=default_values['dry_run'],
                        help='Dry run',)
    parser.add_argument('--stream-id', action='store', type=str,
                        dest='stream_id',
                        default=default_values['stream_id'],
                        metavar='STREAM_ID',
                        help='stream ID',)
    parser.add_argument('--period-frames', action='store', type=int,
                        dest='period_frames',
                        default=default_values['period_frames'],
                        metavar='PERIOD_FRAMES',
                        help='period in frames',)
    parser.add_argument(
            'func', type=str,
            default=default_values['func'],
            choices=FUNC_CHOICES.keys(),
            help='%s' % (' | '.join("{}: {}".format(k, v) for k, v in
                         FUNC_CHOICES.items())),)
    parser.add_argument('infile', type=str,
                        default=default_values['infile'],
                        metavar='input-file',
                        help='input file',)
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
    # do something
    parse_file(options.infile, options)


if __name__ == '__main__':
    # at least the CLI program name: (CLI) execution
    main(sys.argv)
