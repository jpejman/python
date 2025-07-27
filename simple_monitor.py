import argparse
import csv
import curses
import time
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path

DB_PATH = Path('monitor_history.db')

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    ts TEXT PRIMARY KEY,
    cpu_percent REAL,
    mem_used_mb INTEGER,
    mem_total_mb INTEGER,
    disk_used_gb INTEGER,
    disk_total_gb INTEGER,
    net_recv_mb INTEGER,
    net_sent_mb INTEGER
);
"""

class MonitorDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path)
        self.conn.execute(CREATE_TABLE_SQL)
        self.conn.commit()

    def export_csv(self, csv_path):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM metrics ORDER BY ts")
        rows = cur.fetchall()
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([c[0] for c in cur.description])
            writer.writerows(rows)

    def insert_metrics(self, data):
        with self.conn:
            self.conn.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data['ts'],
                    data['cpu_percent'],
                    data['mem_used_mb'],
                    data['mem_total_mb'],
                    data['disk_used_gb'],
                    data['disk_total_gb'],
                    data['net_recv_mb'],
                    data['net_sent_mb'],
                ),
            )

    def close(self):
        self.conn.close()

def read_cpu_times():
    with open('/proc/stat') as f:
        line = f.readline()
    parts = [float(x) for x in line.strip().split()[1:]]
    total = sum(parts)
    idle = parts[3]
    return total, idle


def get_cpu_percent(prev_total, prev_idle, curr_total, curr_idle):
    total_diff = curr_total - prev_total
    idle_diff = curr_idle - prev_idle
    if total_diff == 0:
        return 0.0
    return 100.0 * (1.0 - idle_diff / total_diff)


def read_memory():
    info = {}
    with open('/proc/meminfo') as f:
        for line in f:
            key, value, *_ = line.split()
            info[key.rstrip(':')] = int(value)
    mem_total = info['MemTotal'] // 1024
    mem_available = info.get('MemAvailable', info.get('MemFree', 0)) // 1024
    mem_used = mem_total - mem_available
    return mem_used, mem_total


def read_network():
    with open('/proc/net/dev') as f:
        lines = f.readlines()[2:]
    total_recv = 0
    total_sent = 0
    for line in lines:
        parts = line.split()
        if len(parts) >= 17:
            recv = int(parts[1])
            sent = int(parts[9])
            total_recv += recv
            total_sent += sent
    return total_recv // (1024 * 1024), total_sent // (1024 * 1024)


def read_disk():
    usage = shutil.disk_usage('/')
    return usage.used // (1024 * 1024 * 1024), usage.total // (1024 * 1024 * 1024)


def gather_metrics(prev_cpu_total, prev_cpu_idle, prev_net_recv, prev_net_sent):
    curr_cpu_total, curr_cpu_idle = read_cpu_times()
    cpu_percent = get_cpu_percent(prev_cpu_total, prev_cpu_idle, curr_cpu_total, curr_cpu_idle)

    mem_used, mem_total = read_memory()
    disk_used, disk_total = read_disk()
    net_recv, net_sent = read_network()

    data = {
        'ts': datetime.now().isoformat(timespec='seconds'),
        'cpu_percent': round(cpu_percent, 2),
        'mem_used_mb': mem_used,
        'mem_total_mb': mem_total,
        'disk_used_gb': disk_used,
        'disk_total_gb': disk_total,
        'net_recv_mb': net_recv,
        'net_sent_mb': net_sent,
    }
    return data, curr_cpu_total, curr_cpu_idle, net_recv, net_sent


def draw_screen(stdscr, data):
    stdscr.erase()
    stdscr.addstr(0, 0, 'Simple Monitor')
    stdscr.addstr(2, 0, f"Time: {data['ts']}")
    stdscr.addstr(3, 0, f"CPU Usage: {data['cpu_percent']:.2f}%")
    stdscr.addstr(4, 0, f"Memory: {data['mem_used_mb']}/{data['mem_total_mb']} MB")
    stdscr.addstr(5, 0, f"Disk: {data['disk_used_gb']}/{data['disk_total_gb']} GB")
    stdscr.addstr(6, 0, f"Network (MB): recv {data['net_recv_mb']} | sent {data['net_sent_mb']}")
    stdscr.addstr(8, 0, 'Press q to quit')
    stdscr.refresh()

def run_monitor():
    curses.wrapper(_run_monitor)


def _run_monitor(stdscr):
    curses.curs_set(0)
    db = MonitorDB()
    prev_cpu_total, prev_cpu_idle = read_cpu_times()
    net_recv, net_sent = read_network()

    while True:
        data, prev_cpu_total, prev_cpu_idle, net_recv, net_sent = gather_metrics(
            prev_cpu_total, prev_cpu_idle, net_recv, net_sent
        )
        db.insert_metrics(data)
        draw_screen(stdscr, data)
        stdscr.timeout(1000)
        ch = stdscr.getch()
        if ch == ord('q'):
            break
    db.close()


def export_history(csv_path):
    db = MonitorDB()
    db.export_csv(csv_path)
    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple system monitor')
    parser.add_argument('--export', metavar='CSV', help='export history to CSV file')
    args = parser.parse_args()
    if args.export:
        export_history(args.export)
    else:
        run_monitor()
