
import argparse
import log_parser


parser = argparse.ArgumentParser(description='stretch a log file by copying some lines')
parser.add_argument('logfile', help='log file to stretch')
parser.add_argument('len', type=int, help='total lines the log should have')
ARGS = parser.parse_args()
print("ARGS:", ARGS)

parsed_log = log_parser.parse(ARGS.logfile)

if ARGS.len < len(parsed_log):
    print("log is longer than", ARGS.len)
    exit(1)

factor = ARGS.len / len(parsed_log)
factor -= 1

stretched_name = ARGS.logfile
if stretched_name.endswith('.log'):
    stretched_name = stretched_name[:-4]
stretched_name += '_stretched.log'

with open(ARGS.logfile) as logfile:
    with open(stretched_name, mode='w') as stretched:

        new_len = 0
        count = factor
        for i, line in enumerate(logfile):
            stretched.write(line)
            new_len += 1
            while count >= 1:
                stretched.write(line)
                new_len += 1
                count -= 1
            count += factor

    print("new len:", new_len)
