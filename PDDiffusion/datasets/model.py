"""Dataset index tables"""

from sqlalchemy.orm import registry, relationship
from sqlalchemy import Column, String, Integer, ForeignKey, ForeignKeyConstraint

mapper_registry = registry()
Base = mapper_registry.generate_base()

class Dataset(Base):
    """A source of dataset images.
    
    These are roughly intended to correspond to submodules of the datasets
    module. However, this isn't strictly enforced by the database system, and
    datasets are free to create multiple entries in the same database (for
    example, if the WikimediaCommons package were used to scrape two different
    MediaWiki sites). If your dataset implementation supports multiple
    instances of itself, all IDs it allocates must be prefixed with the name of
    its submodule directory, follwed by a colon.
    
    For example, "WikimediaCommons:https://commons.wikimedia.org/w/api.php"
    
    Datasets are allowed to store additional data in tables outside of the
    ones specified here. These dataset-specific tables are accessed using the
    individual dataset submodules and must specify a one-to-one relationship
    with the source row (e.g. id and dataset_id are foreign keys on the same
    columns in DatasetImage)"""
    __tablename__ = "dataset"

    id = Column(String, primary_key=True)

    images = relationship("DatasetImage", back_populates="dataset")

    def __repr__(self):
        return f"Dataset(id={self.id!r})"

class DatasetImage(Base):
    """A single dataset image.
    
    Each image is a member of a dataset; in fact, image IDs are only unique
    within one dataset.
    
    Each image also owns a File, which specifies where the image itself is
    stored."""
    __tablename__ = "datasetimage"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("dataset.id"), primary_key=True)
    file_id = Column(Integer, ForeignKey("file.id"))

    dataset = relationship("Dataset", back_populates="images")
    file = relationship("File", back_populates="dataset_image")
    labels = relationship("DatasetLabel", back_populates="dataset_image")

    def __repr__(self):
        return f"DatasetImage(id={self.id!r}, dataset_id={self.dataset_id!r})"

class File(Base):
    """A single file.
    
    Files may be stored locally, or on an external server. The storage location
    is determined by the `storage_provider` and `url`. Storage providers are
    defined in the codebase; URLs are used to get the image data from the
    storage provider."""
    __tablename__= "file"

    id = Column(Integer, primary_key=True, autoincrement=True)
    storage_provider = Column(String)
    url = Column(String) #May also be a path

    dataset_image = relationship("DatasetImage", back_populates="file")

    #Indicates a local file. URL is the image filename.
    LOCAL_FILE="local"

    def __repr__(self):
        return f"File(id={self.id!r}, storage_type={self.storage_type!r}, url={self.url!r})"

class DatasetLabel(Base):
    """A single piece of label data associated with an image.
    
    Images may have multiple labels; the data_key field distinguishes betweeen
    the two. The meaning of label keys and values is determined primarily by
    the source dataset, with the one stipulation being that label values must
    be suitable for training CLIP on."""

    __tablename__ = "datasetlabel"
    __table_args__ = (
        ForeignKeyConstraint(["image_id", "dataset_id"], ["datasetimage.id", "datasetimage.dataset_id"]),
    )

    image_id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("dataset.id"), primary_key=True)
    data_key = Column(String, primary_key=True)

    value = Column(String)

    dataset_image = relationship("DatasetImage", back_populates="labels")

    def __repr__(self):
        return f"DatasetLabel(image_id={self.image_id!r}, dataset_id={self.dataset_id!r}, data_key={self.data_key!r}, value={self.value!r})"
