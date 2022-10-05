! pip install -q av

! mkdir "dataset"

! wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
! unrar x -Y "/content/action_recognition_R2plus1d/UCF101.rar" "dataset"

! wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
! unzip '/content/action_recognition_R2plus1d/UCF101TrainTestSplits-RecognitionTask.zip' -d 'dataset'