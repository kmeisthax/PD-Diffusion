"""Utility to install the dataset tables into a database."""

from PDDiffusion.datasets.model import mapper_registry
from argparse_dataclass import dataclass
from dataclasses import field
from sqlalchemy import create_engine
from dotenv import load_dotenv
import sys, os

@dataclass
class SetupOptions:
    verbose: bool = field(default=False, metadata={"args": ["--verbose"], "help": "Log SQL statements as they execute"})

options = SetupOptions.parse_args(sys.argv[1:])
env = load_dotenv()

engine = create_engine(os.getenv("DATABASE_CONNECTION"), echo=options.verbose, future=True)
mapper_registry.metadata.create_all(engine)