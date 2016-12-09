



for var in 2016-01-31--19-19-25.h5 2016-02-02--10-16-58.h5 2016-02-08--14-56-28.h5 2016-02-11--21-32-47.h5 2016-03-29--10-50-20.h5 2016-04-21--14-48-08.h5 2016-05-12--22-20-00.h5 2016-06-02--21-39-29.h5 2016-06-08--11-46-01.h5
do
    unzip comma-dataset.zip camera/$var
    mv camera/$var $var
    unzip comma-dataset.zip log/$var
    mv log/$var log-$var
    cd ../
    python data_helper.py $var
    rm datasets/$var
    rm datasets/log-$var
    cd datasets
    
done

