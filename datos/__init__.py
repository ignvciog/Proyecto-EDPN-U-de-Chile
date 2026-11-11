"""Archivos de datos de volcanes para la actividad 1D."""

from importlib import import_module


def cargar(nombre):
    """Carga `datos.<nombre>` (ej. cargar('calbuco2015'))."""
    return import_module(f"datos.{nombre}")
