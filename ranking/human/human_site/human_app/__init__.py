import django
import logging
from django.conf import settings
from django.core.exceptions import AppRegistryNotReady
from human_site import settings as site_settings

def initialize():
    try:
        new_settings = {key : value for key,value in site_settings.__dict__.items() if key.isupper()}
        settings.configure(**new_settings)
        django.setup()
    except RuntimeError:
        logging.warning("Tried to double configure the API, ignore this if running the Django app directly")

initialize()
try:
    from human_app.models import *
except AppRegistryNotReady:
    logging.info("Could not yet import Feedback")
