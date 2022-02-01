# from django.conf.urls import url
from django.urls import re_path

import backend.consumers as consumers

websocket_urlpatterns = [
    re_path(r"^ws/events", consumers.EventsConsumer.as_asgi()),
]
