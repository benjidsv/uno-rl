import os

from hypothesis import settings

settings.register_profile("heavy", max_examples=1000, deadline=None)
settings.register_profile("default", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
