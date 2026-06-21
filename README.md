# Génération d'images : WGANs_and_SGMs 

Résultats d'un projet académique sur deux méthodes de génération d'image: WGAN (Wasserstein Generative Adversarial Network) et SGM (Score-based Generative Modeling).

Auteurs: Fatoumata NDOUR et Valentin ABONGO


## Contenu du projet

Ce projet est composé d'un rapport contenant la théorie et les remarques sur l'application des deux méthodes de génération d'images. De plus dans le répertoire *notebooks*, il se trouve quatre notebooks Jupyter relatant nos expérimentations.

## Notebooks

Les différents notebooks ont été exécutés sur Google Collab. Donc le code est
adapté à ce cadre. Les notebooks *SGM_circles.ipynb* et *WGAN_circles.ipynb* sont l'application des méthodes respectivement SGM et WGAN pour une base de données synthétique faite d'images de contour blanc de cercle sur fond noir. Les notebooks *SGM_fmnist.ipynb* et *WGAN_fmnist.ipynb* sont l'application des méthodes respectivement SGM et WGAN pour la base de données bien connue Fashion-MNIST faite d'images de prêt-à-porter. Le code est en Python et on présume que le lecteur est familier avec la bibliothèque pytorch. On invite également le lecteur curieux sur les notebooks concernant la méthode SGM de se renseigner sur le package torchdiff (https://torchdiff.readthedocs.io/en/latest/).
