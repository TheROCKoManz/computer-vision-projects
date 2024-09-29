from flask import Flask
from CrowdAnalysis.WebApp_Zone.app import create_zone_blueprint
import os

# Create the main Flask app
main_app = Flask(__name__)
main_app.config['UPLOAD_FOLDER'] = 'Data/Crowd_Count/ZoneCounter_Dynamic/uploads'
os.makedirs(main_app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create and register the Zone blueprint
zone_app = create_zone_blueprint()
main_app.register_blueprint(zone_app, url_prefix='/zone')

if __name__ == '__main__':
    main_app.run(host='0.0.0.0', port=6969, debug=True)
