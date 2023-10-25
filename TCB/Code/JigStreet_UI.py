# %%
import os, sys
import wx
import wx.lib.agw.multidirdialog as MDD
import JigStreet_Rev1 as js
import JigStreet_Rev1_1 as js2

wildcard = "dxf (*.dxf)|*.dxf|" \
            "All files (*.*)|*.*"

# %%
########################################################################
class RedirectText(object):
    def __init__(self,aWxTextCtrl):
        self.out = aWxTextCtrl

    def write(self,string):
        self.out.WriteText(string)
        
class MyForm(wx.Frame):
 
    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY,
                          "MACD Simple Auto Jig Street UI")
        panel = wx.Panel(self, wx.ID_ANY)
        self.currentDirectory = os.getcwd()
        #self.SetSize(700, 700)
        self.Centre()
        
        # create the buttons and bindings
        openFileDlgBtn = wx.Button(panel, label="Select input dxf file")
        openFileDlgBtn.Bind(wx.EVT_BUTTON, self.onOpenFile)
        
        # create input for expand value
        self.label = wx.StaticText(panel, label="Input the desired component margin")
        self.txt = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER, value = "0.65")
        
        #create input for scaling
        self.scaler = wx.CheckBox(panel, label="Scale")
        self.scaler.SetValue(True)
        # self.check_box.Bind(wx.EVT_CHECKBOX, self.on_check_box_change)

        
        # Button to run
        self.runJigraw = wx.Button(panel, label="Run the Script with selected dxf file")
        self.runJigraw.Bind(wx.EVT_BUTTON, self.runscriptraw)
        # self.runJig = wx.Button(panel, label="Run the Script with selected scaled file")
        # self.runJig.Bind(wx.EVT_BUTTON, self.runscript)

        
        # logger
        self.log = wx.TextCtrl(panel, size=(800, 700), style=wx.TE_MULTILINE|wx.TE_READONLY)
        redir = RedirectText(self.log)
        sys.stdout = redir
        
        # put the buttons in a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(openFileDlgBtn, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(self.label, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(self.txt, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(self.scaler, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(self.runJigraw, 1, wx.ALL|wx.CENTER, 5)
        # sizer.Add(self.runJig, 1, wx.ALL|wx.CENTER, 5)

        sizer.Add(self.log, 1, wx.ALL|wx.CENTER, 10)

        # sizer.Add(saveFileDlgBtn, 0, wx.ALL|wx.CENTER, 5)
        # sizer.Add(dirDlgBtn, 0, wx.ALL|wx.CENTER, 5)
        # sizer.Add(multiDirDlgBtn, 0, wx.ALL|wx.CENTER, 5)
        panel.SetSizer(sizer)
        
    #----------------------------------------------------------------------

    def onOpenFile(self, event):
        """
        Create and show the Open FileDialog
        """
        dlg = wx.FileDialog(
            self, message="Choose a file",
            defaultDir=self.currentDirectory, 
            defaultFile="",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
            # style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR
            )
        if dlg.ShowModal() == wx.ID_OK:
            print("You chose the following file:")
            self.path = dlg.GetPath()
            print(self.path)
            # for path in self.paths:
            #     print(path)
        dlg.Destroy()
        
    # #----------------------------------------------------------------------
    def runscript(self,event):
        print(self.path)
        cmargin = float(self.txt.GetValue())
        print(str(cmargin) + " is the component boundary box expand multiplier.")
        js.processjigstreet(self.path, cmargin)
        
    # #----------------------------------------------------------------------
    def runscriptraw(self,event):
        print(self.path)
        cmargin = float(self.txt.GetValue())
        scaletf = self.scaler.GetValue()
        print("Scale is set to :" + str(scaletf))
        print(str(cmargin) + " is the component boundary box expand multiplier.")
        if scaletf:
            js2.processjigstreetraw(self.path, cmargin)
        else:
            js.processjigstreet(self.path,cmargin)
        

        
#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()
# %%
