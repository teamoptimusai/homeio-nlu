#Optimized Model
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1H8lDIH14TGsj_Jl11kvuFOuu9RvZ02FB" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1H8lDIH14TGsj_Jl11kvuFOuu9RvZ02FB" -o epoch50_best_model_trace.pth
rm ./cookie