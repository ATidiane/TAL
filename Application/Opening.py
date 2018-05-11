# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:01:45 2018

@author: 3502264
"""

import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename


class Opening(tkinter.Tk):

    def __init__(self, parent):

        tkinter.Tk.__init__(self, parent)
        self.parent = parent
        self.title = "ExtractMWU"
        self.geometry("300x300")
        self.bgColor = "#880E4F"
        self.bgButton = "#FF9800"
        self.main()
        self['bg'] = self.bgColor

        self.mainloop()

    def destroyFrame(self):
        for widget in self.winfo_children():
            widget.pack_forget()
            widget.place_forget()

    def quitter(self):
        self.destroy()

    def addLabel(self, s, text, side=TOP, font=("ObelixProBroken", 16)):
        l = Label(s, text=text, font=font, padx=10, pady=10,
                  bg=self.bgColor, fg="white")
        l.pack(side=side)
        return l

    def addFrame(self, s, side=TOP):
        return Frame(s, relief=GROOVE, bg=self.bgColor).pack(side=side)

    def addButton(self, s, text, command, width=12,
                  side=TOP, font=("Times", 15)):
        return Button(s, text=text, command=command, font=font, padx=10,
                      pady=10, bg=self.bgButton, width=width).pack(side=side)

    def addButtonPlace(self, s, text, command, width=12, relx=0.5, rely=0.5,
                       anchor=CENTER, font=("Times", 15)):
        b = Button(s, text=text, command=command, font=font, padx=10,
                   pady=10, bg=self.bgButton, width=width)
        return b.place(relx=relx, rely=rely, anchor=anchor)

    def appendText(self, T, text):
        T.insert(END, text, 'color')

    def OpenFile(self):
        fileName = askopenfilename(initialdir=".",
                                   filetypes=(("Text File", "*.txt"),
                                              ("All Files", "*.*")),
                                   title="Choose a file.")
        try:
            self.fichier = open(fileName, 'r')
            self.realWindow()
        except FileNotFoundError:
            print("No such file")

    def realWindow(self):
        self.destroyFrame()
        self.geometry("900x600")
        T = Text(self, height=40, width=70, borderwidth=0)
        scroll = Scrollbar(self, command=T.yview)
        T.configure(yscrollcommand=scroll.set)
        T.tag_configure('color', foreground='#212121',
                        font=('Times', 11, 'normal'))
        scroll.pack(side=RIGHT, fill=Y)
        T.pack(side=RIGHT)
        self.appendText(T, self.fichier.read())

    def main(self):
        self.addLabel(self, "Welcome\nto ExtractMWU")
        self.chooseF = self.addFrame(self, BOTTOM)
        self.addButtonPlace(self.chooseF, "Choose a file",
                            command=self.OpenFile)
        self.addButtonPlace(self.chooseF, "Quit",
                            command=self.quitter, rely=0.8)


Opening(None)
