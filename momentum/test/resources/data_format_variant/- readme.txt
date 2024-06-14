#NOTE: two of the files are intentionally removed from this test set because they are MIPS architecture, which is no longer supported (sgi_int.c3d and sgi_real.c3d)

A set of six C3D files containing data stored as integer and floating-point values in Intel, SGI and DEC endian formats. The files were created using AMASS with Vicon 370 hardware - each file contains identical data, parameters, and header events stored in each of the six basic C3D formats variants and can be used to verify that software applications read the individual C3D formats correctly.

08/10/1999  02:11 PM            43,520 dec_int.c3d
08/10/1999  03:16 PM            80,384 dec_real.c3d
08/10/1999  03:17 PM            43,520 pc_int.c3d
08/10/1999  03:16 PM            80,384 pc_real.c3d
08/10/1999  03:17 PM            43,520 sgi_int.c3d
08/10/1999  03:17 PM            80,384 sgi_real.c3d

Note that the files have FORCE_PLATFORM:ORIGIN information frequently seen in files created by Oxford Metrics/Vicon systems where the force platform origin is stored as a positive value, implying that the mechanical origin of the force plateform is above the surface of the force plate.

The files contain FPLOC, an undocumented parameter group that may be used to compensate for this error when the file is opened by Vicon software but the group function is undocumented.  Since the function is undocumented and not part of the C3D specification this means that the data in the files will be interpretated inaccurately by any normal application that interprets the data.

Another interesting artifact in these files is that the ADC has collected data from 16 channels although only 12 channels (1-6 and 8-14) appear to be connected to the force plates.  This results in "ghost" signals appearing on the unused open ADC inputs.  This does not affect the data in any way but can be confusing to a casual observer who does not realize that the ADC channels are unconnected.

First Parameter        2
Number of Markers    36
Analog Channels        16
First Frame        1
Last Frame        89
Video Sampling Rate    50.00
Analog Sampling Rate    200.00
Scale Factor        0.28
Data Start Record    13
Interpolation Gap    10
C3D File Format        Intel, SGI, and DEC Formats
Data Format        Integer and floating point formats
