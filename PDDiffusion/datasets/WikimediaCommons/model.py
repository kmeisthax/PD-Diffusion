""""""

from PDDiffusion.datasets.model import Base
from sqlalchemy import Column, String, Integer, ForeignKey, ForeignKeyConstraint, JSON, DateTime
from sqlalchemy.orm import relationship

class WikimediaCommonsImage(Base):
    """A dataset image extracted from a Wikimedia Commons File: article."""

    __tablename__ = "wikimediacommons_image"
    __table_args__ = (
        ForeignKeyConstraint(["id", "dataset_id"], ["datasetimage.id", "datasetimage.dataset_id"]),
    )

    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("dataset.id"), primary_key=True)

    post_id = Column(Integer)
    last_edited = Column(DateTime)

    wikidata = Column(JSON)

    base_image = relationship("DatasetImage")

class WikimediaCommonsData(Base):
    """A single piece of text data extracted from Wikimedia Commons wikidata."""

    __tablename__ = "wikimediacommons_data"
    __table_args__ = (
        ForeignKeyConstraint(["image_id", "dataset_id"], ["datasetimage.id", "datasetimage.dataset_id"]),
        ForeignKeyConstraint(["image_id", "dataset_id"], ["wikimediacommons_image.id", "wikimediacommons_image.dataset_id"]),
    )

    image_id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("dataset.id"), primary_key=True)
    data_key = Column(String, primary_key=True)

    value = Column(String)