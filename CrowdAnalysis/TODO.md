# TODO

## Main Pipeline
    Step 1: UI sends video to processor
    Step 2: Processor returns the first frame along with metadata ([width, height]) 
    Step 3: UI displays the frame and enables users to mark vertices of polygons
        Validations:
        0. The UI initiates an empty 3D array in a structure => Zones = [
                                                                 zone_1[[x_1, y_1],...,[x_n, y_n]],
                                                                 zone_2[[x_1, y_1],...,[x_n, y_n]],
                                                                 ...,
                                                                 zone_n[[x_1, y_1],...,[x_n, y]]
                                                                ]
        i. User marks vertices of zones(polygon) on the images, each vertex contains coordinates => (X, Y)
            The coordinate of each vertext should be with reference to the metadata of the image
        ii. User has to mark at least 3 vertices to create a zone
        iii. Upon creation of a zone, the zone should be appended to the 'Zones' list
        iv. Upon clicking 'Create zones and View' button, the 'Zones' list is sent to the processor

    Step 4: The processor uses the zones list to create the zone polygons and returns the frame for the 
            UI to display and the user to verify.

    Step 5: UI provides 2 buttons, one to reset zones and other to start process.
        i. Clicking 'Reset zones' sends the user to the previous page to reselect zones (refer step 3)
        ii. Upon clicking 'start process' the UI creates a streaming connection with the processor and sends a confirmation signal.
    
    
    Step 6: The processor starts the prediction and streams the output frames to the UI over the connection.

    Step 7: After sending the confirmation signal the UI creates a component t0 display an image which takes a state as a prop which is updated whenever the UI recieves a new frame from the processor.
        i. 3 buttons, 'reset zones', 'restart stream', 'select another video'.

## Shubhankar

## Manasij
