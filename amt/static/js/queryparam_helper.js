// ===========================================================
// A bunch of helper functions for an AMT interface 
// ===========================================================

// Note: QS is short for query string, i.e., the <name>=<value> stuff after the
// question mark (?) in the URL.
var NUM_QS_ZEROPAD = 2; // Number of digits for QS parameters

function gup(name) {
    var regexS = "[\\?&]" + name + "=([^&#]*)";
    var regex = new RegExp(regexS);
    var tmpURL = window.location.href;
    var results = regex.exec(tmpURL);
    if (results == null) {
        return "";
    } else {
        return results[1];
    }
}

function decode(strToDecode) {
    return unescape(strToDecode.replace(/\+/g, " "));
}

function get_random_int(min, max) {
  return Math.floor(Math.random() * (max - min)) + min;
}

function zero_pad(num, numZeros) {
    var n = Math.abs(num);
    var zeros = Math.max(0, numZeros - Math.floor(n).toString().length );
    var zeroString = Math.pow(10,zeros).toString().substr(1);
    if (num < 0) {
        zeroString = '-' + zeroString;
    }
    return zeroString+n;
}

function collect_ordered_QS(param_name, pad) {
    var array = []; // Store all the data
    var done = false;
    var i = 1;
    var name = '';
    var val = '';
    while (done == false) {
        name = param_name + zero_pad(i, pad);
        val = decode(gup(name));

        if (val == "") {
            done = true;
        } else {
            array.push(val);
        }
        i += 1;
    }
    return array;
}
