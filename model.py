from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from pygments.lexer import default

db = SQLAlchemy()

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer)
    content = db.Column(db.String, nullable=False)
    is_user = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    session_number = db.Column(db.Integer, nullable=False)
    box_id = db.Column(db.Integer, nullable=False, default = -1)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'content': self.content,
            'is_user': self.is_user,
            'created_at': self.created_at,
            'session_number': self.session_number,
            'box_id': self.box_id
        }
