import os


# Automatically discover the Django application name, such that the module can
# easily be imported by another script or project
def setup():
    module = os.path.split(os.path.dirname(__file__))[-1]
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE", "{}.settings".format(module)
    )
    import django

    django.setup()