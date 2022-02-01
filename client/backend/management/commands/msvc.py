import atexit
import subprocess
import time

from django.conf import settings
from django.core.management.base import BaseCommand


def cleanup(processes):
    timeout_sec = 5
    for process in processes:  # list of your processes
        p_sec = 0
        for _ in range(timeout_sec):
            if process.poll() is None:
                time.sleep(1)
                p_sec += 1
        if p_sec >= timeout_sec:
            process.kill()  # supported from python 2.6
    print("cleaned up!")


# class Command(BaseCommand):
#     def add_arguments(self, parser):
#         # Optional argument
#         parser.add_argument(
#             "--port", type=str, help="Redis port", default="6379",
#         )
#         parser.add_argument(
#             "--host", type=str, help="Redis host", default="localhost",
#         )

#         parser.add_argument(
#             "--daphne_host", type=str, help="Daphne host", default="0.0.0.0",
#         )

#         parser.add_argument(
#             "--daphne_port", type=str, help="Daphne host", default="8003",
#         )

#     def handle(self, *args, **kwargs):
#         self.stdout.write("Starting daphne ...")
#         self.runserver = subprocess.Popen(  # noqa
#             [
#                 f"daphne --port {kwargs['daphne_port']} --bind \
#                 {kwargs['daphne_host']} --proxy-headers SOD.asgi:application"
#             ],
#             shell=True,
#             stdin=subprocess.PIPE,
#             stdout=self.stdout._out,
#             stderr=self.stderr._out,
#         )
#         self.stdout.write("daphne process on %r\n" % self.runserver.pid)

#         # Terminate subprocesses at exit, such that the webserver is not
#         # running indefinately in the the background...
#         atexit.register(cleanup, [self.runserver])

class Command(BaseCommand):
    def add_arguments(self, parser):
        pass

    def handle(self, *args, **kwargs):
        pass

