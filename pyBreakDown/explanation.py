import numpy as np
from collections import deque
from enum import Enum
from recordclass import recordclass
import matplotlib.patches as patches
from matplotlib import pyplot as plt

AttrInfo = recordclass("AttrInfo",["name","value","contribution","cumulative"])

class ExplainerDirection (Enum):
    Up=1
    Down=2

class Explanation:
    """
    Contains algorithm results, including contribiutions of each individual features.
    """
    _INTERCEPT_NAME = "Intercept"
    _INTERCEPT_VALUE = 1

    def __init__ (self, variable_names, variable_values, contributions, direction):
        self._direction = direction
        self._attributes = deque()
        csum = 0
        for (name, value, contribution) in zip(variable_names, variable_values, contributions):
            csum+=contribution
            self._attributes.append(
                AttrInfo(name=name, value=value, contribution=contribution, cumulative=csum)
                )
        self._has_intercept=False
        self._has_final_prognosis=False
    
    def text (self, fwidth=25, contwidth=20, cumulwidth = 20, digits=2):
        """
        Get user-friendly text from of explanation

        Parameters
        ----------
        fwidth : int
            Width of column with feature names, in digits.
        contwidth : int
            Width of column with contributions, in digits.
        cumulwidth : int
            Width of column with cumulative values, in digits.
        digits : int
            Number of decimal places for values.
        """
        if not self._has_intercept or not self._has_final_prognosis:
            return

        lines = [''.join(
                [
                ' = '.join([attr.name, str(attr.value)]).ljust(fwidth), 
                str(round(attr.contribution,digits)).ljust(contwidth), 
                str(round(attr.cumulative, digits)).ljust(cumulwidth)
                ]
                ) for attr in self._attributes]

        print (''.join(
            ["Feature".ljust(fwidth), 
            "Contribution".ljust(contwidth), 
            "Cumulative".ljust(cumulwidth)]))
        print('\n'.join(lines))
        print(''.join(
            ['Final prediction'.ljust(fwidth+contwidth), 
            str(round(self._final_prediction, digits)).ljust(cumulwidth)]))
        print(' = '.join(["Baseline", str(round(self._baseline, digits))]))

    def visualize(self, figsize=(7,6), filename=None, dpi=90,fontsize=14):
        """
        Get user friendly visualization of explanation

        Parameters
        ----------
        figsize : tuple int
            Pyplot figure size
        filename : string
            Name of file to save the visualization. 
            If not specified, standard pyplot.show() will be performed.
        dpi : int
            Digits per inch for saving to the file
        """

        if not self._has_intercept or not self._has_final_prognosis:
            return

        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
        positions = list(range(len(self._attributes)+2))
        previous_value = self._baseline
        for (attr_info, position) in zip(self._attributes, positions[1:]):
            cumulative = attr_info.cumulative+self._baseline
            height=1
            left = previous_value if attr_info.contribution > 0 else cumulative
            width = abs(attr_info.contribution)
            color = "blue" if attr_info.contribution > 0 else "orange"
            rect = patches.Rectangle(
                xy=(left, position-0.5),width=width,height=height,alpha=0.8,color=color)
            ax.add_patch(rect)
            plt.errorbar(x=left, y=position, yerr=0.5, color="black")
            plt.errorbar(x=left+width, y=position, yerr=0.5, color="black")
            plt.text(left+width+0.15, y=position-0.2, size=fontsize,
                     s = self._get_prefix(attr_info.contribution) + str(round(attr_info.contribution,2)))
            previous_value = cumulative
        
        #add final prediction bar
        rectf = patches.Rectangle(
            xy=(self._baseline,positions[len(positions)-1]-0.5), 
            width=self._final_prediction, 
            height=1, color="grey", alpha=0.8
        )
        ax.add_patch(rectf)
        ax.axvline(x=self._baseline,mew=3,color="black",alpha=1)
        plt.errorbar(x=self._baseline, y=len(positions)-1, yerr=0.5, color="black")
        plt.errorbar(x=self._baseline+self._final_prediction, y=len(positions)-1, yerr=0.5, color="black")
        plt.text(
            x=self._baseline+self._final_prediction+0.15,
            y=positions[len(positions)-1]-0.2,
            s=str(round(self._final_prediction+self._baseline,2)),size=fontsize,weight="bold")

        ax.set_yticks(positions[1:])
        ax.grid(color="gray",alpha=0.5)
        sign = "+" if self._direction==ExplainerDirection.Up else "-"
        labels=[sign + "=".join([attr.name,str(attr.value)]) for attr in self._attributes]+["Final Prognosis"]
        ax.set_yticklabels(labels,size=fontsize)
        
        all_cumulative = [attr.cumulative for attr in self._attributes]
        leftbound = min([min(all_cumulative), 0]) + self._baseline
        rightbound= max(max(all_cumulative)+self._baseline,self._baseline)
        plt.text(x=self._baseline+0.15, y=positions[0]-0.2, s="Baseline = "+str(round(self._baseline,2)),
                size=fontsize,color="red")

        ax.set_xlim(leftbound-1, rightbound+1)
        ax.set_ylim(-1,len(self._attributes)+2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        approach = "\"up\"" if self._direction==ExplainerDirection.Up else "\"down\""
        plt.title("Prediction explanation for "+approach+" approach")

        #fig.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        #fig.subplots_adjust(hspace=0, wspace=0.1)
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename,dpi=dpi)

    def add_intercept (self, intercept_contribution):
        self._attributes.appendleft(AttrInfo(
            name=self._INTERCEPT_NAME, 
            value=self._INTERCEPT_VALUE, 
            contribution=intercept_contribution, 
            cumulative=0)
            )
        self._correct_cumulatives()
        self._has_intercept = True

    def make_final_prediction (self):
        self._final_prediction = sum(attr.contribution for attr in self._attributes)
        self._has_final_prognosis = True

    def add_baseline (self, baseline):
        self._baseline = baseline

    def _correct_cumulatives(self):
        csum = 0
        for attribute in self._attributes:
            csum+=attribute.contribution
            attribute.cumulative = csum

    def _get_prefix(self, val):
        return "+" if val>=0 else ""