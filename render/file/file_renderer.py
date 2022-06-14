from abc import ABC

from render.renderer import Renderer


class FileRenderer(Renderer, ABC):
    def __init__(self, filepath: str):
        self._filepath = filepath
