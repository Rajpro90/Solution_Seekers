import os
import sys
import subprocess
import webbrowser
import time
from threading import Timer

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Checking and installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def init_db():
    """Initialize the database if it doesn't exist."""
    db_path = os.path.join("backend", "disaster_db.sqlite")
    if not os.path.exists(db_path):
        print("Database not found. Initializing...")
        # We can trigger the DB creation by importing the app and db
        # However, it's safer to run it as a subprocess to avoid import issues in this script
        # For now, let's rely on the backend app to create it on startup as per __init__.py logic
        print(f"Database will be created at {db_path} when the backend starts.")
    else:
        print(f"Database found at {db_path}.")

def open_browser():
    """Open the frontend in the default browser."""
    print(f"Opening app: http://localhost:5000")
    webbrowser.open("http://localhost:5000")

def run_backend():
    """Run the Flask backend."""
    print("Starting Flask backend...")
    backend_script = os.path.join("backend", "run.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("backend") # Ensure backend directory is in path
    
    # Open browser after a slight delay to ensure server starts
    Timer(3, open_browser).start()

    try:
        subprocess.check_call([sys.executable, backend_script], env=env)
    except KeyboardInterrupt:
        print("\nStopping backend...")

def kill_port(port):
    """Kill any process listening on the specified port (Windows only)."""
    print(f"Checking for processes on port {port}...")
    try:
        # Find the PID
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        lines = output.strip().split('\n')
        pids = set()
        for line in lines:
            parts = line.split()
            # In netstat -ano, PID is the last column
            if len(parts) > 4:
                pid = parts[-1] 
                if pid != '0': # Don't kill system idle process
                    pids.add(pid)
        
        for pid in pids:
            print(f"Killing process {pid} on port {port}...")
            # /F force, /PID pid
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=False)
            
    except subprocess.CalledProcessError:
        # No process found or error running netstat (expected if port is free)
        pass 
    except Exception as e:
        print(f"Warning: Could not cleanup port {port}: {e}")

if __name__ == "__main__":
    # Ensure we are in the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    install_dependencies()
    # Ensure port 5000 is free
    kill_port(5000)
    init_db()
    run_backend()
