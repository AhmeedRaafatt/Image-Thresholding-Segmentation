import sys
from PyQt6.QtWidgets import QApplication
from MainWindowUI import MainWindowUI
import asyncio
from qasync import QEventLoop
from AgglomerativeKMeans import AgglomerativeKMeans
from RegionGrowingMeanShift import AgglomerativeMeanShift
from Thresholding import Thresholding

async def main():
    # Initialize classes
    agglomerative_kmeans = AgglomerativeKMeans()
    agglomerative_mean_shift = AgglomerativeMeanShift()
    thresholding = Thresholding()
    ### 
    
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    
    window = MainWindowUI()
    window.show()
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)