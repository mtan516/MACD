# %%
import os
import wx
import wx.lib.agw.multidirdialog as MDD
import JigStreet_Rev1 as js

wildcard = "dxf (*.dxf)|*.dxf|" \
            "All files (*.*)|*.*"

# %%
########################################################################
class MyForm(wx.Frame):
 
    #----------------------------------------------------------------------
    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY,
                          "MACD Simple Auto Jig Street UI")
        panel = wx.Panel(self, wx.ID_ANY)
        self.currentDirectory = os.getcwd()
        self.Centre()
        
        # create the buttons and bindings
        openFileDlgBtn = wx.Button(panel, label="Select input dxf file")
        openFileDlgBtn.Bind(wx.EVT_BUTTON, self.onOpenFile)
        
        # create input for expand value
        self.label = wx.StaticText(panel, label="Input the desired component margin")
        self.txt = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER, value = "0.65")
        
        # Button to run
        self.runJig = wx.Button(panel, label="Run the Script with selected file")
        self.runJig.Bind(wx.EVT_BUTTON, self.runscript)
        
        # Button to export
        # self.expJig = wx.Button(panel, label="Export to dxf")
        # self.runJig.Bind(wx.EVT_BUTTON, self.exportdxf)
        
        # put the buttons in a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(openFileDlgBtn, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(self.label, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(self.txt, 0, wx.ALL|wx.CENTER, 5)
        sizer.Add(self.runJig, 1, wx.ALL|wx.CENTER, 5)

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
            self.paths = dlg.GetPaths()
            print("You chose the following file(s):")
            for path in self.paths:
                print(path)
        dlg.Destroy()
        print("Add script execution step here next")
        print(str(self.txt.GetValue()))
        
    # #----------------------------------------------------------------------
    def runscript(self,event):
        print(self.paths)
        cmargin = float(self.txt.GetValue())
        print(str(cmargin))
        js.processjigstreet(self.paths[0], cmargin)
        
    # #----------------------------------------------------------------------
    
        

        
#----------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    app = wx.App(False)
    frame = MyForm()
    frame.Show()
    app.MainLoop()
# %%
