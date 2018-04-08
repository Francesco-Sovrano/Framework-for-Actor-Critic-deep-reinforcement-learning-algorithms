
import datetime


def parse(log_fname):

    log = []

    with open(log_fname) as logfile:
        for i, line in enumerate(logfile):
            try:
                splitted = line.split(' ')

                date_str = splitted[0] + ' ' + splitted[1]
                date = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')

                obj = {'date': date}

                for x in splitted[2:]:
                    if '=' in x:
                        key, val = x.split('=')
                        obj[key] = float(val)
                log.append(obj)

            except Exception as e:
                print("exc %s on line %s" % (repr(e), i+1))
                print("skipping line")
                continue

    return log
