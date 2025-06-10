wsd_english_eval_2017="WSD_Evaluation_Framework"
wget -O "${wsd_english_eval_2017}.zip" http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
unzip "${wsd_english_eval_2017}.zip"
rm "${wsd_english_eval_2017}.zip"

# The data has come from this website https://sapienzanlp.github.io/xl-wsd/docs/data/
# which is where you can find the original Google drive link
xl_wsd_2021="xl-wsd"
wget -O "${xl_wsd_2021}.zip" "https://drive.usercontent.google.com/download?id=19YTL-Uq95hjiFZfgwEpXRgcYGCR_PQY0&export=download&confirm=xxx"
unzip "${xl_wsd_2021}.zip"
rm "${xl_wsd_2021}.zip"