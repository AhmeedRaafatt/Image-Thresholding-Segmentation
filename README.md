# Image-Thresholding-Segmentation

The project features a PyQt-based UI with interactive tabs for Thresholding and Segmentation, exploring methods to separate objects from backgrounds or cluster pixels into meaningful regions.

## Thresholding

Techniques to convert grayscale images into binary images using optimal threshold values.

| Method                  | Image                                      | Description                                      |
|-------------------------|--------------------------------------------|--------------------------------------------------|
| Otsu-Global-Thresholding | ![Otsu-Global-Thresholding](https://github.com/user-attachments/assets/d6ae4ec2-1045-4da0-a8c0-4112ab993962) | Minimizes intra-class variance globally.         |
| Otsu-Local-Thresholding  | ![Otsu-Local-Thresholding](https://github.com/user-attachments/assets/df1bf433-4251-4c2d-a5ee-704f2522b6d6) | Minimizes intra-class variance locally.          |
| Optimal-Global-Thresholding | ![Optimal-Global-Thresholding](https://github.com/user-attachments/assets/58f86c18-8be9-4d95-be7c-621eacaa551e) | Maximizes inter-class variance globally.    |
| Optimal-Local-Thresholding | ![Optimal-Local-Thresholding](https://github.com/user-attachments/assets/2662c40d-ce04-42e0-9363-055ab6c9044c) | Maximizes inter-class variance locally.     |
| Spectral-Local-Thresholding | ![Spectral-Local-Thresholding](https://github.com/user-attachments/assets/c7a1aba4-2c08-4acb-a861-abe1a599ed77) | Applies frequency-based thresholding locally. |
| Spectral-Global-Thresholding | ![Spectral-Global-Thresholding](https://github.com/user-attachments/assets/493eb093-200e-425b-b6b2-65675ddbdb87) | Applies frequency-based thresholding globally. |

## Segmentation

Methods to group pixels into regions based on similarity.

| Method            | Image                                      | Description                                      |
|-------------------|--------------------------------------------|--------------------------------------------------|
| K-Means-Clustering | ![K-Means-Clustering](https://github.com/user-attachments/assets/01ace9f1-1775-4886-bc6d-4a82647ed7bb) | Clusters pixels by assigning to k centers.       |
| Agglomerative-Clustering | ![Agglomerative-Clustering](https://github.com/user-attachments/assets/142a8bc3-d769-4b80-94ec-fa7f7dd1d229) | Merges pixels using average linkage.         |
| Mean-Shift-Clustering | ![Mean-Shift-Clustering](https://github.com/user-attachments/assets/f0900670-9a48-4662-b4a6-4a624b5a2eb6) | Computes window means, optimized with Cython. |
| Region-Growing-Segmentation | ![Region-Growing-Segmentation](https://github.com/user-attachments/assets/b887f50e-10e0-44c1-ba3a-d89172ec5c33) | Grows regions from seed points using BFS.   |

## Getting Started

1. Clone the repository: `git clone https://github.com/Ayatullah-ahmed/Image-Thresholding-Segmentation.git`
2. Install dependencies (e.g., PyQt, NumPy, Cython).
3. Run the UI to explore the techniques.

## Contributors

<table>
  <tr>
        <td align="center">
      <a href="https://github.com/salahmohamed03">
        <img src="https://avatars.githubusercontent.com/u/93553073?v=4" width="250px;" alt="Salah Mohamed"/>
        <br />
        <sub><b>Salah Mohamed</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Ayatullah-ahmed" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/125223938?v=" width="250px;" alt="Ayatullah Ahmed"/>
        <br />
        <sub><b>Ayatullah Ahmed</b></sub>
      </a>
    </td>
        <td align="center">
      <a href="https://github.com/Abdelrahman0Sayed">
        <img src="https://avatars.githubusercontent.com/u/113141265?v=4" width="250px;" alt="Abdelrahman Sayed"/>
        <br />
        <sub><b>Abdelrahman Sayed</b></sub>
      </a>
    </td>
        </td>
        <td align="center">
      <a href="https://github.com/AhmeedRaafatt">
        <img src="https://avatars.githubusercontent.com/u/125607744?v=4" width="250px;" alt="Ahmed Raffat"/>
        <br />
        <sub><b>Ahmed Rafaat</b></sub>
      </a>
    </td>
  </tr>
</table>
