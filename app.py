from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from datetime import datetime, timedelta
from new_rag import send_continue_chat
from model import db, ChatHistory
from dotenv import load_dotenv
import ssl
import os

app = Flask(__name__)
CORS(app)  # Enable CORS
socketio = SocketIO(app, cors_allowed_origins='*')

# Load environment variables from .env file
load_dotenv()

# Database connection parameters
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'sslmode': os.getenv('DB_SSLMODE'),
    'sslrootcert': os.getenv('DB_SSLROOTCERT')
}
# SSL context
ssl_context = ssl.create_default_context(cafile=db_params['sslrootcert'])


## Configure PostgreSQL with parameters and SSL
app.config['SQLALCHEMY_DATABASE_URI'] = (
    f"postgresql://{db_params['user']}:{db_params['password']}@"
    f"{db_params['host']}:{db_params['port']}/{db_params['dbname']}?sslmode={db_params['sslmode']}&sslrootcert={db_params['sslrootcert']}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)  # Initialize the database with the app


def create_response(data, status_code, message):
    return jsonify({
        'data': data,
        'statusCode': status_code,
        'message': message
    }), status_code

def get_last_session_number(user_id):
    last_session = db.session.query(ChatHistory.session_number).filter_by(user_id=user_id).order_by(ChatHistory.session_number.desc()).first()
    if last_session is None:
        return 0
    return last_session[0]


def get_new_session_number(user_id):
    last_session_number = get_last_session_number(user_id)
    last_session_records = db.session.query(ChatHistory).filter_by(user_id=user_id, session_number=last_session_number).all()

    if not last_session_records:
        return last_session_number + 1

    last_record_time = last_session_records[-1].created_at
    current_time = datetime.utcnow()
    time_diff = current_time - last_record_time

    if time_diff > timedelta(minutes=5) or len(last_session_records) > 15:
        return last_session_number + 1

    return last_session_number

def get_chat_history_by_session(user_id, session_number):
    return db.session.query(ChatHistory).filter_by(user_id=user_id, session_number=session_number).all()

@app.route('/chat', methods=['POST'])
def send_message():
    try:
        data = request.json
        user_query = data.get('content')
        user_id = data.get('userId')

        if user_query and user_id:

            # Get old session number
            last_session_number = get_last_session_number(user_id)
            # Get the new session number
            session_number = get_new_session_number(user_id)

            if session_number > last_session_number:
                answer = send_continue_chat([], user_query)
            else:
                # Get chat history of the new session
                chat_history = get_chat_history_by_session(user_id, session_number)
                if chat_history:
                    answer = send_continue_chat(chat_history, user_query)
                else:
                    answer = send_continue_chat([], user_query)

            # Save the user query to the chat history
            user_chat_history = ChatHistory(
                user_id=user_id,
                content=user_query,
                is_user=True,
                session_number=session_number
            )
            db.session.add(user_chat_history)

            # Save the generated answer to the chat history
            bot_chat_history = ChatHistory(
                user_id=user_id,
                content=answer,
                is_user=False,
                session_number=session_number
            )
            db.session.add(bot_chat_history)

            db.session.commit()

            return create_response(bot_chat_history.to_dict(), 200, 'Success')
        return create_response(None, 400, 'No query or userId provided')
    except Exception as e:
        print(e)
        return create_response(None, 500, f'Internal Server Error: {str(e)}')

@app.route('/chat/all', methods=['GET'])
def get_all_chat_history():
    try:
        chat_history_records = ChatHistory.query.all()
        chat_history_list = [record.to_dict() for record in chat_history_records]
        return create_response(chat_history_list, 200, 'Success')
    except Exception as e:
        return create_response(None, 500, f'Internal Server Error: {str(e)}')

@app.route('/chat/<int:user_id>', methods=['GET'])
def get_chat_history_by_user(user_id):
    try:
        chat_history_records = ChatHistory.query.filter_by(user_id=user_id).all()
        chat_history_list = [record.to_dict() for record in chat_history_records]
        return create_response(chat_history_list, 200, 'Success')
    except Exception as e:
        return create_response(None, 500, f'Internal Server Error: {str(e)}')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=4646)
