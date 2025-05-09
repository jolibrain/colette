import threading
from collections import OrderedDict


class SessionCache:
    def __init__(self, max_sessions=100):
        self.sessions = OrderedDict()
        self.max_sessions = max_sessions
        self.lock = threading.Lock()  # To handle concurrent access if necessary

    def create_session(self, session_id):
        with self.lock:
            if session_id not in self.sessions:
                self._ensure_capacity()
                self.sessions[session_id] = []
                self.sessions.move_to_end(session_id)

    def get_session(self, session_id):
        with self.lock:
            if session_id in self.sessions:
                # Move the session to the end to mark it as recently used
                self.sessions.move_to_end(session_id)
                return self.sessions[session_id]
            return None

    def update_session(self, session_id, message):
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].append(message)
                self.sessions.move_to_end(session_id)
            else:
                # If the session does not exist, create it
                self.create_session(session_id)
                self.sessions[session_id].append(message)

    def update_streaming(self, session_id, response):
        with self.lock:
            self.sessions[session_id][-1]["content"][-1]["content"] = response

    def get_message(self, session_id, position):
        with self.lock:
            if session_id in self.sessions and 0 <= position < len(
                self.sessions[session_id]
            ):
                return self.sessions[session_id][position]
            return None

    def session_exists(self, session_id):
        with self.lock:
            return session_id in self.sessions

    def _ensure_capacity(self):
        # Ensure that the number of sessions does not exceed the max_sessions limit
        if len(self.sessions) >= self.max_sessions:
            # Pop the first item (i.e., the oldest session)
            self.sessions.popitem(last=False)
