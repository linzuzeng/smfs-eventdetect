pragma rtGlobals=1		// Use modern global access method.


Menu "Git"
"ExportTorontoFile"
End

Function ExportTorontoFile()

Variable iii=1
for(iii=0;iii<900;iii+=1)
String extend
if (iii < 10)
	extend = num2str(0) + num2str(0) + num2str(0)
elseif (iii < 100)
	extend = num2str(0) + num2str(0)
elseif (iii < 1000)
	extend = num2str(0)
else
	extend = ""
	
endif

Save/T/P=home root:constSpeed:$("Tension_Unf"+num2str(iii)) as  "Tension_Unf"+extend+num2str(iii)+".itx"
Save/T/P=home root:constSpeed:$("Tension_Ref"+num2str(iii)) as  "Tension_Ref"+extend+num2str(iii)+".itx"
Save/T/P=home root:constSpeed:$("Distance_Unf"+num2str(iii)) as  "Distance_Unf"+extend+num2str(iii)+".itx"
Save/T/P=home root:constSpeed:$("Distance_Ref"+num2str(iii)) as  "Distance_Ref"+extend+num2str(iii)+".itx"

endfor
End

End
