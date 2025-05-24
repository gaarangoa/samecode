import sys
import time

class ProgressBar:
    def __init__(self, total, prefix='', width=40, fill='█', print_end='\r'):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.fill = fill
        self.print_end = print_end
        self.iteration = 0
        self.start_time = time.time()

    def update(self, step=1, **info):
        self.iteration += step
        percent = self.iteration / self.total
        filled_length = int(self.width * percent)
        bar = self.fill * filled_length + '-' * (self.width - filled_length)
        elapsed = time.time() - self.start_time

        # Create info string from passed keyword arguments
        info_str = ' '.join(f'{k}={v}' for k, v in info.items())

        # Print formatted progress bar with info on the left
        sys.stdout.write(
            f'\r{self.prefix} {info_str} |{bar}| {self.iteration}/{self.total} '
            f'({percent*100:.1f}%) Elapsed: {elapsed:.1f}s'
        )
        sys.stdout.flush()

        if self.iteration >= self.total:
            sys.stdout.write('\n')

    def reset(self):
        self.iteration = 0
        self.start_time = time.time()
