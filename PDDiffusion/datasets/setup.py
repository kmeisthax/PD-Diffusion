"""Utility to install the dataset tables into a database."""

from PDDiffusion.datasets.model import mapper_registry
from argparse_dataclass import dataclass
from dataclasses import field
from sqlalchemy import create_engine
import sys

@dataclass
class SetupOptions:
    connection_string: str = field(metadata={"args": ["--url"], "help": "Connection string for the DB server or file to use"})
    verbose: bool = field(default=False, metadata={"args": ["--verbose"], "help": "Log SQL statements as they execute"})

options = SetupOptions.parse_args(sys.argv[1:])
engine = create_engine(options.connection_string, echo=options.verbose, future=True)
mapper_registry.metadata.create_all(engine)