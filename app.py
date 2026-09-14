import os
import sys
import runpy

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

runpy.run_path(os.path.join(root_dir, "phicnet", "app.py"), run_name="__main__")
