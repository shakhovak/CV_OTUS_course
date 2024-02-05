# ДЗ 1
**Цель:**
Собрать свой docker контейнер для работы с моделями глубокого обучения.

## Описание/Пошаговая инструкция выполнения домашнего задания:

- [x] Установить и настроить Docker.
- [x] Если используете NVIDIA GPU, то также установить nvidia-docker.
- [x] Собрать и запустить контейнер с Ubuntu 18.04 (или выше), развернуть в контейнере PyTorch, TensorFlow, JupyterLab, OpenCV.
- [x] Зайдите в Google Colab и запросите GPU.
- [x] В качестве результатов отправьте в личный кабинет следующее:
  - Конфигурацию вашего компьютера: ОС, модель GPU и CPU.
  - Скриншот или лог, который будет содержать версию PyTorch и TensorFlow, запущенные из под Docker.
  - Скриншот или лог, который будет содержать версию PyTorch и TensorFlow, запущенные в Google Colab.
<hr>

## Решение:
Начальная конфигурация Windows + установленный Docker Dekstop, поэтому все команды для докера работают в cmd. Собираю контейнер командой ```docker build -t otus-cuda:1.0.0 .``` 
Установленная ОС + процессор
![image](https://github.com/shakhovak/CV_OTUS_course/assets/89096305/a3516f1b-4e15-4b4b-9663-38eeb1634979)
![image](https://github.com/shakhovak/CV_OTUS_course/assets/89096305/a9644f81-3642-4845-b3fd-f84e9be7dc43)

Запускаю контейнер командой:  ```docker run -it --name otus --volume=/home/data:/playground/data -p 8789:8789 --gpus all --rm otus-test:1.0.0```

Появляется ссылка на jupyter lab, в нем запускаю проверку версий torch + tensorflow, проверяю наличие GPU

![image](https://github.com/shakhovak/CV_OTUS_course/assets/89096305/0fee1462-af89-45e9-87c4-6cf5982b9ed8)

Потом можно будет запускать отдельно ноутбук из контейнера для выполнения заданий командой ```jupyter notebook --no-browser --port=8789 --ip=0.0.0.0 --allow-root```

Такой же запрос в  google colab:

![image](https://github.com/shakhovak/CV_OTUS_course/assets/89096305/9f8177ff-10bf-42b6-b1d9-34f6b20ddfe2)



