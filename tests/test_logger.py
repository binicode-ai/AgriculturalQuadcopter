import numpy as np

from payload.logger import FlightLogger

logger = FlightLogger()

for i in range(5):

    logger.log(

        image_id=i,

        timestamp=i,

        position=np.array([

            i,

            2*i,

            15

        ]),

        attitude=np.array([

            0,

            0,

            0.1*i

        ]),

        velocity=np.array([

            3,

            0,

            0

        ]),

        waypoint=1

    )

logger.save_csv()