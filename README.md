# HW02 - wildcatshawkeyes
 
# HW02Q7
 In order to execute the code you must run  
 git clone https://github.com/wildcatshawkeyesgh/hw02wildcatshawkeyes  
 then run  
 uv sync  
 then you must create a directory for the media at the top level run  
 mkdir media  
 then run  
 uv run /scripts/binaryclassification_animate_impl.py  

 The location of the video will be located in the media folder  

# HW02Q
In order to execute the code to make the boxplot you must run git clone 
https://github.com/wildcatshawkeyesgh/hw02wildcatshawkeyes

then run  
uv sync  
then run  

mkdir data  

then you must make both shell scripts executable   
chmod +x malwaredatadownload.sh  
chmod +x multiclass_impl.sh  
then run the shell script  
./malwaredatadownload.sh  
now you need to edit multiclass_impl.sh where the default input location is to be your directory location of where you stored the malware data csv  
as well as the the eval script will be looking in a data directory you need to set that location to be the same data directory as for the malware.  
now you can run and generate the boxplots.  
./multiclass_impl.sh  
The boxplot should be in the data folder  
 That should work I know my code can run once and work I know It can run for a little while without failing when trying to run it 5 times but I ran out of time to see if ita actual works when trying to run it 5 times. The only thing I am   unsure about is if my boxplot will actually generate.  
