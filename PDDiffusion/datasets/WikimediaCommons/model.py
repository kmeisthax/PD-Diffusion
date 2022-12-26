""""""

from PDDiffusion.datasets.model import Base, DatasetImage
from sqlalchemy import Column, String, Integer, ForeignKey, ForeignKeyConstraint, JSON, DateTime, select
from sqlalchemy.orm import relationship

BASE_API_ENDPOINT = "https://commons.wikimedia.org/w/api.php"

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
    
    def select_all_image_articles(session):
        """Given a database session, select all the articles that are also images.
        
        Returns an iterable that yields both the dataset image and the Wikimedia
        Commons article data in a tuple. Order is article, image."""
        return session.execute(
            select(WikimediaCommonsImage, DatasetImage)
                .join_from(WikimediaCommonsImage, DatasetImage, WikimediaCommonsImage.base_image)
                .where(DatasetImage.dataset_id == f"WikimediaCommons:{BASE_API_ENDPOINT}", DatasetImage.is_banned == False)
        )
