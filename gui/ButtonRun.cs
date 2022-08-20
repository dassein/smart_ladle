using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.ComponentModel;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Events;
using UnityEditor;
using UnityEngine.UI;
using TMPro;

using System.IO;
using System.Text;
// Require support of EventTrigger script
[RequireComponent(typeof(UnityEngine.EventSystems.EventTrigger))]
public class ButtonRun : MonoBehaviour
{
    EventTrigger trigger;
    public GameObject loadingPanel;


    // Start is called before the first frame update
    void Start()
    {
        Button btn = this.GetComponent<Button>();
        //trigger = GetComponent<EventTrigger>(); // add trigger to the button run
        trigger = btn.gameObject.GetComponent<EventTrigger>();
        trigger.triggers = new List<EventTrigger.Entry>(); // initialize EventTrigger.Entry container
        EventTrigger.Entry enter = new EventTrigger.Entry(); //instantiate EventTrigger.Entry object
        enter.eventID = EventTriggerType.PointerClick; // specify trigger type: click
        enter.callback = new EventTrigger.TriggerEvent(); // what to do after being triggered
        trigger.triggers.Add(enter);
        UnityAction<BaseEventData> action = new UnityAction<BaseEventData>(ClickButton); // function
        enter.callback.AddListener(action);
    }

    // Update is called once per frame
    void Update()
    {

    }
    void ClickButton(BaseEventData pd)
    {
        //StartCoroutine(ButtonClickSubroutine());
        /* extract the argumetns from inputfield: heat no, ladle no, caster no */
        string heatNo = GameObject.Find("InputHeatNo").GetComponent<InputField>().text;
        string ladleNo = GameObject.Find("InputLadleNo").GetComponent<InputField>().text;
        string locationCode = GameObject.Find("InputLocationCode1").GetComponent<TMP_Dropdown>().captionText.text;
        Image outputImage = GameObject.Find("IMGButton").GetComponent<Image>();
        bool isHeatNoNumeric = int.TryParse(heatNo, out _); // see https://stackoverflow.com/questions/894263/identify-if-a-string-is-a-number

        if (((!isHeatNoNumeric) && (heatNo != ""))) //|| ((!isLadleNoNumeric) && (ladleNo != "")) || (!isLocationCodeNumeric))
        {
            return;
        }
        else if ((heatNo == "") && (ladleNo == ""))
        {
            return;
        }
        string args; // string args = "10 15"; for debug
        if (heatNo == "")
        {
            args = "--ladle_no "; //+ ladleNo + " --location_code " + locationCode;
        }
        else
        {
            args = "--heat_no " + heatNo; //+ " --location_code " + locationCode;
        }

        bool isPrediction = true; // Run button means prediction
        string direction = "0"; // for prediction, run direction is zero

        /* execute the program */
        // argument format: heatNo casterNo prediction? direction
        args = heatNo + " " + locationCode + " " + isPrediction + " " + direction;

        string exeFilename = "smart-ladle.exe";

        string workDir = Application.dataPath + "/Resources/SmartLadle/";
        bool successRun = false;

        successRun = ExecuteProgram(exeFilename, workDir, args);

        if (!successRun)
        {
            return;
        }

        string csvPath = Application.dataPath + "/Resources/SmartLadle/output.csv";
        Dictionary<string, string> paramDict = new Dictionary<string, string>();
        paramDict = GetParamDict(csvPath);

        if (locationCode == "1")
        {

            Sprite caster1Image = Resources.Load<Sprite>("SmartLadle/output1");
            outputImage.sprite = caster1Image;

        }
        else
        {

            Sprite caster2Image = Resources.Load<Sprite>("SmartLadle/output2");
            outputImage.sprite = caster2Image;

        }

    }



    //private IEnumerator ButtonClickSubroutine()
    //{
    //    ToggleLoading();
    //    yield return new WaitForEndOfFrame();

    //    /* extract the argumetns from inputfield: heat no, ladle no, caster no */
    //    string heatNo = GameObject.Find("InputHeatNo").GetComponent<InputField>().text;
    //    string ladleNo = GameObject.Find("InputLadleNo").GetComponent<InputField>().text;
    //    string locationCode = GameObject.Find("InputLocationCode1").GetComponent<TMP_Dropdown>().captionText.text;
    //    Image outputImage = GameObject.Find("IMGButton").GetComponent<Image>();
    //    bool isHeatNoNumeric = int.TryParse(heatNo, out _); // see https://stackoverflow.com/questions/894263/identify-if-a-string-is-a-number

    //    if (((!isHeatNoNumeric) && (heatNo != ""))) //|| ((!isLadleNoNumeric) && (ladleNo != "")) || (!isLocationCodeNumeric))
    //    {
    //        yield return null;
    //    }
    //    else if ((heatNo == "") && (ladleNo == ""))
    //    {
    //        yield return null;
    //    }
    //    string args; // string args = "10 15"; for debug
    //    if (heatNo == "")
    //    {
    //        args = "--ladle_no "; //+ ladleNo + " --location_code " + locationCode;
    //    }
    //    else
    //    {
    //        args = "--heat_no " + heatNo; //+ " --location_code " + locationCode;
    //    }

    //    bool isPrediction = true; // Run button means prediction
    //    string direction = "0"; // for prediction, run direction is zero

    //    /* execute the program */
    //    // argument format: heatNo casterNo prediction? direction
    //    args = heatNo + " " + locationCode + " " + isPrediction + " " + direction;

    //    string exeFilename = "smart_simple.exe";

    //    string workDir = Application.dataPath + "/Resources/SmartLadle/";
    //    bool successRun = false;

    //    successRun = ExecuteProgram(exeFilename, workDir, args);

    //    if (!successRun)
    //    {
    //        yield return null;
    //    }

    //    string csvPath = Application.dataPath + "/Resources/SmartLadle/output.csv";
    //    Dictionary<string, string> paramDict = new Dictionary<string, string>();
    //    paramDict = GetParamDict(csvPath);

    //    if (locationCode == "1")
    //    {

    //        Sprite caster1Image = Resources.Load<Sprite>("SmartLadle/output1");
    //        outputImage.sprite = caster1Image;

    //    }
    //    else
    //    {

    //        Sprite caster2Image = Resources.Load<Sprite>("SmartLadle/output2");
    //        outputImage.sprite = caster2Image;

    //    }

    //    ToggleLoading();

    //}

    //private void ToggleLoading()
    //{

    //    loadingPanel.SetActive(!loadingPanel.activeSelf);

    //}


    static bool ExecuteProgram(string exeFilename, string workDir, string args)
    {

        bool rt = true;

        string tempString = workDir + exeFilename;

        //tempString.Replace('/', '\\');

        UnityEngine.Debug.Log(tempString);

        ProcessStartInfo processInfo = new ProcessStartInfo(tempString , args);
        processInfo.CreateNoWindow = true;
        processInfo.UseShellExecute = true;

        Process runningEXE = Process.Start(processInfo);
        runningEXE.WaitForExit();

        return rt;
    }


    static Dictionary<string, string> GetParamDict(string csvPath)
    {
        string[][] array2D;
        List<string> tempList1 = new List<string>();
        List<string> tempList2 = new List<string>();
        DataHandler scriptholderDataHandler = GameObject.Find("Scriptholder").GetComponent<DataHandler>();
        int dropdownReference = GameObject.Find("InputLocationCode1").GetComponent<TMP_Dropdown>().value + 1;
        //TextAsset binAsset = Resources.Load(csvPath, typeof(TextAsset)) as TextAsset; // load csv as binary 
        // string[] lineArray = binAsset.text.Split("\r"[0]); // split lines with "\r"
        string content = File.ReadAllText(csvPath, Encoding.Default);// load csv
        string[] lineArray = content.Split(new string[] { "\r\n" }, StringSplitOptions.RemoveEmptyEntries); // split lines with "\r\n"
        array2D = new string[lineArray.Length][]; 
        for (int i = 0; i < lineArray.Length; i++)         // parse elements for each row
        {
            array2D[i] = lineArray[i].Split(','); // split columns with ","
        }
        Dictionary<string, string> paramDict = new Dictionary<string, string>();
        int lenColumn = array2D[0].Length;

        for (int j = 0; j < lenColumn; j++)
        {
            paramDict.Add(array2D[0][j], array2D[dropdownReference][j]); // store 0-th row as key, 1-st row as value
            //UnityEngine.Debug.Log(array2D[dropdownReference][j].ToString());
            tempList1.Add(array2D[1][j]);
            tempList2.Add(array2D[2][j]);
        }

        scriptholderDataHandler.UpdateCasterValues(tempList1, tempList2);

        return paramDict;
    }
}
