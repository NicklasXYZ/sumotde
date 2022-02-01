import os
import asyncio 
import shutil
import json
import logging
from tempfile import TemporaryDirectory
from typing import Optional, Union, List, Dict, Any, Tuple


from channels.generic.websocket import AsyncJsonWebsocketConsumer
# from channels.db import database_sync_to_async
# from django.apps import apps
# from django.conf import settings
# from django.db import models
# from django.db.utils import DatabaseError

from backend.utils import (
    ansi_escape,
)

# Local type alias
Events = List[Dict[str, Any]]

# Parameters & settings
logging.basicConfig(level = logging.DEBUG)

PROJECT_NAME = 'project'
PROJECT_FILE = f'{PROJECT_NAME}/src/{PROJECT_NAME}.gleam'


async def run_subprocess(
    commandline_args: str,
    cwd: Union[None, str],
    ) -> Tuple[List[str], List[str], int]:
    """Run shell commands.
    Args:
        commandline_args (str): Shell commands to be run.
        cwd (Union[None, str], optional): The current working directory. 
            Defaults to None.
    Returns:
        Tuple[List[str], List[str], int]: stdout, stderror and a return code
    """
    logging.debug('Subprocess commandline args: ' + commandline_args)
    s = await asyncio.create_subprocess_shell(
        commandline_args,
        stdout = asyncio.subprocess.PIPE,
        stderr = asyncio.subprocess.STDOUT,
        cwd = cwd,
    )
    stdout_data, stderr_data = await s.communicate()
    logging.debug('\nSubprocess stdout: ')
    for string in stdout_data.decode('utf-8').split('\n'):
        logging.debug(string)
    logging.debug('\nSubprocess stderr: ')
    for string in str(stderr_data).split('\n'):
        logging.debug(string)
    # Check the returncode to see whether the process terminated normally
    if s.returncode == 0:
        logging.debug(
            'Subprocess exited normally with return code: ' + str(s.returncode)
            )
        return stdout_data.decode('utf-8').split('\n'), str(stderr_data).split('\n'), 0
    else:
        logging.debug(
            'Subprocess exited with non-zero return code: ' + str(s.returncode)
        )
        return stdout_data.decode('utf-8').split('\n'), str(stderr_data).split('\n'), -1


async def handle_output(
    stdout: List[str],
    stderr: List[str],
    ) -> Events:
    """Organize stdout and stderr for a frontend application to consume.
    Args:
        stdout (List[str]): Standard output resulting from running a command in
            a shell.
        stderr (List[str]): Error output resulting from running a command in 
            a shell.
    Returns:
        Events: Organized stdout and stderr output.
    """
    events = []
    events.extend([
            {
                'Message': ansi_escape.sub('', _),
                'Kind': 'stdout',
                # NOTE: Delay is currently not used
                'Delay': 0,
            } for _ in stdout
        ]
    )
    if stderr is None:
        events.append(
            {
                'Message': ansi_escape.sub('', stderr),
                'Kind': 'stderr',
                # NOTE: Delay is currently not used
                'Delay': 0,
            }
        )
    return events


# async def run(message: Dict[str, Any]):
#     result = await request.json()
#     events = []; formatted = None
#     # Create a temporary directory for compiling and running a Gleam snippet
#     with TemporaryDirectory() as td:
#         # Copy a default and pre-defined Gleam project to the temporary directory
#         shutil.copytree(
#             f'./{PROJECT_NAME}',
#             f'{td}/{PROJECT_NAME}',
#             copy_function = shutil.copy,
#         )
#         # Check that everything was copied properly to the temporary directory
#         if os.path.exists(f'{td}/{PROJECT_NAME}'):
#             # Write the Gleam code snippet we would like to run to a file in the
#             # default Gleam project  
#             with open(f'{td}/{PROJECT_FILE}', 'w') as f:
#                 f.write(result['code'])
#             # Compile the given Gleam code snippet
#             stdout, stderr, rc = await run_subprocess(
#                 f'export HOME={td} && rebar3 escriptize',
#                 cwd = f'{td}/{PROJECT_NAME}'
#             )
#             # Save all events from stdout and stderr such that we can forward these to 
#             # the user in the frontend
#             events_ = await handle_output(stdout, stderr)
#             events.extend(events_)
#             # If the Gleam project was compilled successfully then try to run the code
#             if rc == 0:
#                 stdout, stderr, rc = await run_subprocess(
#                     f'_build/default/bin/{PROJECT_NAME}',
#                     cwd = f'{td}/{PROJECT_NAME}',
#                 )
#                 # Again, save all events from stdout and stderr such that we can forward
#                 # these to the user in the frontend
#                 events_ = await handle_output(stdout, stderr)
#                 events.extend(events_)
#         # ... Else raise an exception and log the attempt
#         else:
#             logging.debug('The scenario could not be run...')
#             logging.debug(f'Temp dir: {td}')
#     # Return associated events (stdout and stderr)
#     response = {'events': events}
#     return response



async def run(message: Dict[str, Any]):
    stdout, stderr, rc = await run_subprocess(
        f'sumo --help',
        cwd = None,
    )
    events_ = await handle_output(stdout, stderr)
    for item in events_:
        print(item)
    print("rc    : ", rc)
    return stdout, stderr, rc


class EventsConsumer(AsyncJsonWebsocketConsumer):


    async def connect(self):
        # Accept the connection
        await self.accept()
        # Send messages on connect
        await self.on_ws_connect()

    async def receive(self, text_data):
        # Parse the received text data
        received_json_data = json.loads(text_data)
        print("Data: ", received_json_data)

        # Determine the message type
        message_type = received_json_data["type"].strip().lower()
        if "data" in received_json_data:
            data = received_json_data["data"]
            # Respond to the received message
            await self.respond(message_type, data)
        else:
            pass

    #
    # Message handlers
    #

    async def on_ws_connect(self):
        message = {
            "type": "websocket-connect",
            "data": None,
        }
        logging.debug("Websocket connection established.")
        await self.send_json(message)

    async def on_ws_disconnect(self):
        message = {
            "type": "websocket-disconnect",
            "data": None,
        }
        logging.debug("Websocket connection terminated.")
        await self.send_json(message)

    async def echo_message(self, event):
        data = event["data"]
        message = {
            "type": "echo-message",
            "data": data,
        }
        if "error" in event:
            message["error"] = event["error"]
        await self.send_json(message)

    async def respond(self, message_type, data):
        if message_type == "run-simulator":
            await run(data)


