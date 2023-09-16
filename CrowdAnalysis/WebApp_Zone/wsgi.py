import sys
sys.path.append('CrowdAnalysis/WebApp_Zone/')
from app import app

if __name__ == '__main__':
    app.run(debug=True)