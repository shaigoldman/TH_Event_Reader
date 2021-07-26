# TH_Event_Reader

This git package will read TH events with their pathInfo.

Currently cmlreaders cannot read pathInfo, so I created this. It also includes some other
helpful pathinfo like baseline start and stop ms times.

To install:
    
    pip install git+https://github.com/shaigoldman/TH_Event_Reader.git

To use in code, first install TH_EventReader from the
package th_eventreader. Then you can read TH events! Example:

    from th_eventreader import TH_EventReader as TReader
    events = TReader.get_events(subj='R1076D', montage=0, session=0, exp='TH1')