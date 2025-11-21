Prototipo de Interacción Humano-Máquina usando Lenguaje Corporal.
=============

Innovation Fest 2025

Este repositorio utiliza la implementación original [Mario-Level-1](https://github.com/justinmeister/Mario-Level-1) de [justinmeister](https://github.com/justinmeister) como motor de juego para simular el primer nivel del juego clásico de Mario. 

El código permite jugar el nivel usando expresiones corporales para sustituir el uso de botones: 

* Mano Derecha: Controla el avance retroceso del personaje. 
* Mano Izquierda: Controla los saltos del personaje. 
* Pierna derecha: Controla el botón de acción.

## Instalación: 
El juego requiere de la instalación de Python 3 y las siguientes librerías: 

```
pip3 install opencv-python
pip3 install numpy
pip3 install pygame
pip3 install mediapipe
```

## Uso: 
Primero, ejecutar el modelo de detección de Pose:

```
python run_model.py
```

Posteriormente, ejecutar el juego:

```
python mario_level_1.py
```


**DISCLAIMER:**

Este proyecto está desarrollado únicamente para fines educativos y de aprendizaje.
