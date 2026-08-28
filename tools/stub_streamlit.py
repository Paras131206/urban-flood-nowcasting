"""A fake Streamlit, so the pages can be executed without a browser.

Why this exists
---------------
The three pages are the largest and least tested part of this project. A typo
in a dictionary key or a variable used before it is set does not show up until
someone clicks the tab it lives in — which, at a demo, is the worst possible
moment to find out.

This module installs stand-ins for streamlit, folium, streamlit_folium and
streamlit_autorefresh into sys.modules, then a page can simply be exec'd.
Every widget returns its default value, so the run takes the default path
through the page; layout calls record themselves and do nothing.

It cannot catch a visual problem, and it does not pretend to be Streamlit.
What it catches is the class of bug that actually bites: NameError, KeyError,
a column list that does not match the DataFrame, a function called with
arguments it does not have.

Run it with tools/check_pages.py.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager


class _Recorder:
    """Accepts any call, records it, and returns something plausible."""

    def __init__(self, log, overrides=None):
        self.log = log
        # Label substring -> value, so a scenario can steer the page down a
        # branch the defaults never reach. Without this the emergency routing
        # profile and the high-tide case are never executed at all.
        self.overrides = overrides if overrides is not None else {}

    def _override(self, label):
        # Exact labels only. Substring matching looked convenient and then
        # matched "Rainfall" against a slider called "Rainfall (mm/hr)",
        # handing a string to float() and reporting a harness bug as a page
        # crash. A scenario that names a label wrongly should quietly do
        # nothing, not silently steer the wrong widget.
        if str(label) in self.overrides:
            return True, self.overrides[str(label)]
        return False, None

    # -- layout ----------------------------------------------------------- #
    def columns(self, spec, **kw):
        count = spec if isinstance(spec, int) else len(spec)
        return [self for _ in range(count)]

    def tabs(self, labels, **kw):
        return [self for _ in labels]

    @contextmanager
    def expander(self, label, **kw):
        self.log.append(("expander", label))
        yield self

    @contextmanager
    def spinner(self, text="", **kw):
        yield

    @contextmanager
    def container(self, *a, **kw):
        yield self

    @contextmanager
    def form(self, *a, **kw):
        yield self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # -- widgets: every one returns its default --------------------------- #
    def selectbox(self, label, options, index=0, format_func=None, **kw):
        options = list(options)
        if not options:
            return None
        forced, value = self._override(label)
        choice = value if forced and value in options else \
            options[min(index or 0, len(options) - 1)]
        if format_func:
            for option in options:
                format_func(option)             # exercise every label it makes
        return choice

    def radio(self, label, options, index=0, format_func=None, **kw):
        return self.selectbox(label, options, index, format_func, **kw)

    def slider(self, label, min_value=0, max_value=100, value=None, **kw):
        forced, forced_value = self._override(label)
        if forced:
            return forced_value
        return value if value is not None else min_value

    def select_slider(self, label, options=(), value=None, format_func=None, **kw):
        options = list(options)
        if format_func:
            for option in options:
                format_func(option)             # exercise every branch
        forced, forced_value = self._override(label)
        if forced and forced_value in options:
            return forced_value
        if value is not None:
            return value
        return options[0] if options else None

    def checkbox(self, label, value=False, **kw):
        forced, forced_value = self._override(label)
        return forced_value if forced else value

    def toggle(self, label, value=False, **kw):
        return value

    def button(self, label, **kw):
        return False                            # never "clicked"

    def file_uploader(self, label, **kw):
        return None

    def text_input(self, label, value="", **kw):
        return value

    def number_input(self, label, value=0, **kw):
        return value

    # -- output: record and move on --------------------------------------- #
    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append((name, args[0] if args else None))
            return None
        return call


class _CacheData:
    """@st.cache_data, with and without arguments. Just calls through."""

    def __call__(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def wrap(fn):
            return fn
        return wrap

    def clear(self):
        pass


def install(overrides=None) -> list:
    """Put the fakes into sys.modules. Returns the call log."""
    log: list = []
    st = _Recorder(log, overrides or {})

    module = types.ModuleType("streamlit")
    for attribute in dir(_Recorder):
        if not attribute.startswith("_"):
            setattr(module, attribute, getattr(st, attribute))
    module.cache_data = _CacheData()
    module.cache_resource = _CacheData()
    module.session_state = {}
    module.sidebar = st
    module.stop = lambda: (_ for _ in ()).throw(_Stop())
    module.__getattr__ = lambda name: getattr(st, name)

    components = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = lambda *a, **k: None
    v1.iframe = lambda *a, **k: None
    components.v1 = v1
    module.components = components

    sys.modules["streamlit"] = module
    sys.modules["streamlit.components"] = components
    sys.modules["streamlit.components.v1"] = v1

    # folium: every constructor returns something that accepts add_to()
    folium = types.ModuleType("folium")

    class _Layer:
        def __init__(self, *a, **k):
            pass

        def add_to(self, other):
            return self

        def add_child(self, other):
            return self

    for name in ("Map", "PolyLine", "CircleMarker", "Marker", "Tooltip",
                 "DivIcon", "FeatureGroup", "LayerControl", "Popup", "Icon"):
        setattr(folium, name, type(name, (_Layer,), {}))
    sys.modules["folium"] = folium

    sf = types.ModuleType("streamlit_folium")
    sf.st_folium = lambda *a, **k: {}
    sf.folium_static = lambda *a, **k: None
    sys.modules["streamlit_folium"] = sf

    ar = types.ModuleType("streamlit_autorefresh")
    ar.st_autorefresh = lambda *a, **k: 0
    sys.modules["streamlit_autorefresh"] = ar

    return log


class _Stop(Exception):
    """Raised by st.stop(), which is control flow rather than an error."""


Stop = _Stop
