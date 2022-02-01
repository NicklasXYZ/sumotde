from channels.routing import ProtocolTypeRouter, URLRouter

import backend.routing as routing

application = ProtocolTypeRouter(
    {
        "websocket": URLRouter(routing.websocket_urlpatterns,),
    }
)
