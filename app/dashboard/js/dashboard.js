d3.json(data_loc, function (error, data) {
    var metadata = data;
    _.each(metadata, function(d) {

    // ===================================================

        d.MS_ID = +d.MS_ID;
        d.Messages = +d.Messages;
        d.Seconds = +d.Seconds/10;
        d.Hour = +d.Hour;
        d.SentimentFirst = +d.SentimentFirst;
        d.SentimentLast = +d.SentimentLast;
        d.SentimentFit = +d.SentimentFit;
        d.Date = d3.time.week(new Date(d.Date));
    });

    // ===================================================

    var ndx = crossfilter(metadata);
    var AgentIDDim  = ndx.dimension(dc.pluck('AgentID'));
    var TopicDim  = ndx.dimension(dc.pluck('Topic'));
    
    var DateDim  = ndx.dimension(dc.pluck('Date'));
    var MonthDim  = ndx.dimension(dc.pluck('Month'));
    var DayDim  = ndx.dimension(dc.pluck('Day'));
    var HourRange = [.95, 54.6];
    var HourBinWidth = (HourRange[1]-HourRange[0])/24;
    var HourDim = ndx.dimension(function(d) {
        var HourThresholded = d.Hour;
        return HourBinWidth * Math.floor(HourThresholded / HourBinWidth);
    });
    
    var MessagesRange = [1, 22];
    var MessagesBinWidth = (MessagesRange[1]-MessagesRange[0])/20;
    var MessagesDim = ndx.dimension(function(d) {
        var MessagesThresholded = d.Messages;
        return MessagesBinWidth * Math.floor(MessagesThresholded / MessagesBinWidth);
    });
    
    var SecondsRange = [4, 562];
    var SecondsBinWidth = (SecondsRange[1]-SecondsRange[0])/20;
    var SecondsDim = ndx.dimension(function(d) {
        var SecondsThresholded = d.Seconds;
        return SecondsBinWidth * Math.floor(SecondsThresholded / SecondsBinWidth);
    });
    
    var SentimentFirstRange = [-1.05, 1.05];
    var SentimentFirstBinWidth = (SentimentFirstRange[1]-SentimentFirstRange[0])/20;
    var SentimentFirstDim = ndx.dimension(function(d) {
        var SentimentFirstThresholded = d.SentimentFirst;
        return SentimentFirstBinWidth * Math.floor(SentimentFirstThresholded / SentimentFirstBinWidth);
    });
    
    var SentimentLastRange = [-1.05, 1.05];
    var SentimentLastBinWidth = (SentimentLastRange[1]-SentimentLastRange[0])/20;
    var SentimentLastDim = ndx.dimension(function(d) {
        var SentimentLastThresholded = d.SentimentLast;
        return SentimentLastBinWidth * Math.floor(SentimentLastThresholded / SentimentLastBinWidth);
    });
    
    var SentimentFitRange = [-1.05, 1.05];
    var SentimentFitBinWidth = (SentimentFitRange[1]-SentimentFitRange[0])/20;
    var SentimentFitDim = ndx.dimension(function(d) {
        var SentimentFitThresholded = d.SentimentFit;
        return SentimentFitBinWidth * Math.floor(SentimentFitThresholded / SentimentFitBinWidth);
    });

    var allDim = ndx.dimension(function(d) {return d;});

    // ===================================================

    var countPerAgentID = AgentIDDim.group().reduceCount();
    var countPerTopic = TopicDim.group().reduceCount();

    var countPerDate = DateDim.group().reduceSum(function (d) {
        return 1;
    });
    var countPerMonth = MonthDim.group().reduceCount();
    var countPerDay = DayDim.group().reduceCount();
    var countPerHour = HourDim.group().reduceCount();

    var countPerMessages = MessagesDim.group().reduceCount();
    var countPerSeconds = SecondsDim.group().reduceCount();

    var countPerSentimentFirst = SentimentFirstDim.group().reduceCount();
    var countPerSentimentLast = SentimentLastDim.group().reduceCount();
    var countPerSentimentFit = SentimentFitDim.group().reduceCount();

    var all = ndx.groupAll();

    // ===================================================
    var AgentIDChart = dc.pieChart('#chart-AgentID');
    var TopicChart = dc.rowChart('#chart-Topic');

    var DateChart = dc.barChart('#chart-Date');
    var MonthChart = dc.pieChart('#chart-Month');
    var DayChart = dc.pieChart('#chart-Day');
    var HourChart = dc.barChart('#chart-Hour');

    var MessagesChart  = dc.barChart('#chart-Messages');
    var SecondsChart  = dc.barChart('#chart-Seconds');

    var SentimentFirstChart  = dc.barChart('#chart-SentimentFirst');
    var SentimentLastChart  = dc.barChart('#chart-SentimentLast');
    var SentimentFitChart  = dc.barChart('#chart-SentimentFit');

    var dataCount = dc.dataCount('#data-count');
    var dataTable = dc.dataTable('#data-table');

    // ===================================================

    var TopicColor = d3.scale.category20();
    TopicChart
        .width(450)
        .height(540)
        .dimension(TopicDim)
        .group(countPerTopic)
        .elasticX(true)
        .margins({top: 10, right: 10, bottom: 20, left: 10})
        .colors(function (d) {return TopicColor(d);});

    var MonthColor = d3.scale.category20();
    MonthChart
        .width(d3.select('#chart-Month').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(MonthDim)
        .group(countPerMonth)
        .colors(function (d) {return MonthColor(d);})
        .innerRadius(30)
        .ordering(function (d) {
            var order = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            };
            return order[d.key];
        });

    var DayColor = d3.scale.category20();
    DayChart
        .width(d3.select('#chart-Day').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(DayDim)
        .group(countPerDay)
        .colors(function (d) {return DayColor(d);})
        .innerRadius(30)
        .ordering(function (d) {
            var order = {
                'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,
                'Fri': 4, 'Sat': 5, 'Sun': 6
            };
            return order[d.key];
        });

    var AgentIDColor = d3.scale.category10();
    AgentIDChart
        .width(d3.select('#chart-AgentID').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(AgentIDDim)
        .group(countPerAgentID)
        .colors(function (d) {return AgentIDColor(d);})
        .innerRadius(30);

    // ===================================================

    // ===================================================

    var minDate = new Date('2016-04-01 06:00:56');
    var maxDate = new Date('2016-06-02 14:09:41');
    DateChart
        .width(d3.select('#chart-Date').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(DateDim)
        .group(countPerDate)
        .elasticY(true)
        .barPadding(1)
        .renderHorizontalGridLines(true)
        .margins({top: 10, right: 10, bottom: 20, left: 60})
        .x(d3.time.scale().domain([new Date(minDate), new Date(maxDate)]))
        .round(d3.time.week.round)
        .alwaysUseRounding(true)
        .xUnits(d3.time.weeks);

    var minHour = 0;
    var maxHour = 24.5;
    HourChart
        .width(d3.select('#chart-Hour').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(HourDim)
        .group(countPerHour)
        .elasticY(true)
        .barPadding(1)
        .renderHorizontalGridLines(true)
        .margins({top: 10, right: 10, bottom: 20, left: 60})
        .x(d3.scale.linear().domain([minHour, maxHour]))
        .xUnits(dc.units.fp.precision(HourBinWidth))
        .round(function(d) {
            return HourBinWidth * Math.floor(d / HourBinWidth)
        });

    var minSentimentFirst = -1.05;
    var maxSentimentFirst = 1.05;
    SentimentFirstChart
        .width(d3.select('#chart-SentimentFirst').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(SentimentFirstDim)
        .group(countPerSentimentFirst)
        .elasticY(true)
        .barPadding(1)
        .renderHorizontalGridLines(true)
        .margins({top: 10, right: 10, bottom: 20, left: 60})
        .x(d3.scale.linear().domain([minSentimentFirst, maxSentimentFirst]))
        .xUnits(dc.units.fp.precision(SentimentFirstBinWidth))
        .round(function(d) {
            return SentimentFirstBinWidth * Math.floor(d / SentimentFirstBinWidth)
        });
    
    var minSentimentLast = -1.05;
    var maxSentimentLast = 1.05;
    SentimentLastChart
        .width(d3.select('#chart-SentimentLast').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(SentimentLastDim)
        .group(countPerSentimentLast)
        .elasticY(true)
        .barPadding(1)
        .renderHorizontalGridLines(true)
        .margins({top: 10, right: 10, bottom: 20, left: 60})
        .x(d3.scale.linear().domain([minSentimentLast, maxSentimentLast]))
        .xUnits(dc.units.fp.precision(SentimentLastBinWidth))
        .round(function(d) {
            return SentimentLastBinWidth * Math.floor(d / SentimentLastBinWidth)
        });
    
    var minSentimentFit = -1.05;
    var maxSentimentFit = 1.05;
    SentimentFitChart
        .width(d3.select('#chart-SentimentFit').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(SentimentFitDim)
        .group(countPerSentimentFit)
        .elasticY(true)
        .barPadding(1)
        .renderHorizontalGridLines(true)
        .margins({top: 10, right: 10, bottom: 20, left: 60})
        .x(d3.scale.linear().domain([minSentimentFit, maxSentimentFit]))
        .xUnits(dc.units.fp.precision(SentimentFitBinWidth))
        .round(function(d) {
            return SentimentFitBinWidth * Math.floor(d / SentimentFitBinWidth)
        });
    
    var minMessages = .95;
    var maxMessages = 22;
    MessagesChart
        .width(d3.select('#chart-Messages').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(MessagesDim)
        .group(countPerMessages)
        .elasticY(true)
        .barPadding(1)
        .renderHorizontalGridLines(true)
        .margins({top: 10, right: 10, bottom: 20, left: 60})
        .x(d3.scale.linear().domain([minMessages, maxMessages]))
        .xUnits(dc.units.fp.precision(MessagesBinWidth))
        .round(function(d) {
            return MessagesBinWidth * Math.floor(d / MessagesBinWidth)
        });

    var minSeconds = 4;
    var maxSeconds = 562;
    SecondsChart
        .width(d3.select('#chart-Seconds').node().parentNode.getBoundingClientRect().width * 0.9)
        .height(240)
        .dimension(SecondsDim)
        .group(countPerSeconds)
        .elasticY(true)
        .barPadding(1)
        .renderHorizontalGridLines(true)
        .margins({top: 10, right: 10, bottom: 20, left: 60})
        .x(d3.scale.linear().domain([minSeconds, maxSeconds]))
        .xUnits(dc.units.fp.precision(SecondsBinWidth))
        .round(function(d) {
            return SecondsBinWidth * Math.floor(d / SecondsBinWidth)
        });

    // ===================================================

    dataCount
        .dimension(ndx)
        .group(all);

    dataTable
        .dimension(allDim)
        .group(function (d) { return '';})
        .size(200)
        .columns([
            function (d) { return d.MS_ID; },
            function (d) { return d.AgentID; },
            function (d) { return d.Topic; },
            function (d) { return d.Messages; },
            function (d) { return d.Seconds; },
            function (d) { return d.Date; },
            function (d) { return d.Month; },
            function (d) { return d.Day; },
            function (d) { return d.Hour; },
            function (d) { return d.SentimentFirst; },
            function (d) { return d.SentimentLast; },
            function (d) { return d.SentimentFit; },
        ])
        .sortBy(dc.pluck('MS_ID'))
        .order(d3.ascending)
        .on('renderlet', function (table) {
            table.select('tr.dc-table-group').remove();
        });

    // ===================================================

    // ===================================================

    d3.selectAll('a#all').on('click', function () {
        dc.filterAll();
        dc.renderAll();
    });
    d3.selectAll('a#AgentID').on('click', function () {
        AgentIDChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#Topic').on('click', function () {
        TopicChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#Messages').on('click', function () {
        MessagesChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#Seconds').on('click', function () {
        SecondsChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#Date').on('click', function () {
        DateChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#Month').on('click', function () {
        MonthChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#Day').on('click', function () {
        DayChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#Hour').on('click', function () {
        HourChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#SentimentFirst').on('click', function () {
        SentimentFirstChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#SentimentLast').on('click', function () {
        SentimentLastChart.filterAll();
        dc.redrawAll();
    });
    d3.selectAll('a#SentimentFit').on('click', function () {
        SentimentFitChart.filterAll();
        dc.redrawAll();
    });

    dc.renderAll();

    // ===================================================

    var resizeTimer;
    $(window).on('resize', function(e) {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            AgentIDChart
                .width(d3.select('#chart-AgentID').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .redraw();
            TopicChart
                .width(d3.select('#chart-Topic').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(540)
                .redraw();
            MessagesChart
                .width(d3.select('#chart-Messages').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .rescale()
                .redraw();
            SecondsChart
                .width(d3.select('#chart-Seconds').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .rescale()
                .redraw();
            DateChart
                .width(d3.select('#chart-Date').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .rescale()
                .redraw();
            MonthChart
                .width(d3.select('#chart-Month').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .redraw();
            DayChart
                .width(d3.select('#chart-Day').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .redraw();
            HourChart
                .width(d3.select('#chart-Hour').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .rescale()
                .redraw();
            SentimentFirstChart
                .width(d3.select('#chart-SentimentFirst').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .rescale()
                .redraw();
            SentimentLastChart
                .width(d3.select('#chart-SentimentLast').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .rescale()
                .redraw();
            SentimentFitChart
                .width(d3.select('#chart-SentimentFit').node().parentNode.getBoundingClientRect().width * 0.9)
                .height(240)
                .rescale()
                .redraw();
        }, 100);
    });
});