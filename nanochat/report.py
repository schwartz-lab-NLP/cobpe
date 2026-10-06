"""Write training and evaluation summaries alongside experiment outputs."""

import argparse
import datetime
import re
from pathlib import Path

from nanochat.common import get_base_dir, get_dist_info


class Report:
    def __init__(self, report_dir):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def log(self, section, data):
        slug = re.sub(r'[^a-z0-9]+', '-', section.lower()).strip('-')
        path = self.report_dir / f'{slug}.md'
        lines = [f'## {section}', '', f'UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}', '']
        for item in data:
            if not item:
                continue
            if isinstance(item, str):
                lines.append(item)
            else:
                lines.extend(f'- {key}: {value}' for key, value in item.items())
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return str(path)

    def generate(self):
        path = self.report_dir / 'report.md'
        sections = sorted(p for p in self.report_dir.glob('*.md') if p != path)
        path.write_text('# CoBPE experiment report\n\n' + '\n'.join(p.read_text(encoding='utf-8') for p in sections), encoding='utf-8')
        return str(path)


class DummyReport:
    def log(self, section, data):
        return None


def get_report():
    _, rank, _, _ = get_dist_info()
    return Report(Path(get_base_dir()) / 'report') if rank == 0 else DummyReport()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true', help='Combine summary sections into report.md')
    args = parser.parse_args()
    if args.generate:
        print(get_report().generate())
