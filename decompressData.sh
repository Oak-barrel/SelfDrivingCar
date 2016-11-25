



for var in 2016-01-30--13-46-00.h5
do
    unzip comma-dataset.zip camera/$var
    mv camera/$var ../datasets/$var
    unzip comma-dataset.zip log/$var
    mv log/$var ../datasets/log-$var
    cd ../SelfDrivingCar
    python data_helper.py $var
    rm ../datasets/$var
    rm ../datasets/log-$var
    cd ../research
done

