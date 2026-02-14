-- Upstream
CREATE TABLE IF NOT EXISTS TeamMembers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS Projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    owner_id INTEGER NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES TeamMembers (id)
);

CREATE TABLE IF NOT EXISTS Tasks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    assignee_id INTEGER,
    due_date TEXT,
    FOREIGN KEY (project_id) REFERENCES Projects (id),
    FOREIGN KEY (assignee_id) REFERENCES TeamMembers (id)
);

CREATE TABLE IF NOT EXISTS Comments (
    id INTEGER PRIMARY KEY,
    task_id INTEGER NOT NULL,
    message TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES Tasks (id),
    FOREIGN KEY (user_id) REFERENCES TeamMembers (id)
);

-- Downstream
DROP TABLE IF EXISTS Comments;
DROP TABLE IF EXISTS Tasks;
DROP TABLE IF EXISTS Projects;
DROP TABLE IF EXISTS TeamMembers;


