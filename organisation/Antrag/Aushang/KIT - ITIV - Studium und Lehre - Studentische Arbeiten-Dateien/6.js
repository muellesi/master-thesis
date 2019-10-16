if (typeof jQuery == "function") {
    $(document).ready( function () {

	copyrightset = []
	$("[copyright]").each(
		function() {
			copyrighttext = $(this).attr("copyright")
			copyrighttext = copyrighttext.replace(/\(c\)/gi, '&copy;')
			copyrighttext = copyrighttext.replace(/©/g, '&copy;')
			// copyrighttext = copyrighttext.replace(/Foto/g, '')
			// copyrighttext = copyrighttext.replace(/^[\s:]+|[\s:]+$/gm,'')
			if (copyrighttext != "") {
				imageurl = ""
				if ($(this).prop('nodeName') == "IMG") {
					imageurl = $(this).attr("src")
				}
				else {
					imageurl = $(this).css("background-image")
					imageurl = imageurl.replace('url(','').replace(')','').replace(/\"/gi, "");
				}
				if (imageurl != "") {
					if (!Array.isArray(copyrightset[copyrighttext])) copyrightset[copyrighttext] = []
					copyrightset[copyrighttext].push(imageurl)
				}
			}
		}
	)
	copyrighttext = ""
    Object.keys(copyrightset).forEach(function(name, index) {
		copyrighttext += "<p>" + name + ":<br>"
		for (var i=0; i<copyrightset[name].length; i++) {
			copyrighttext += "<a href=\"" + copyrightset[name][i] + "\"><img src=\"" + copyrightset[name][i] + "\" style=\"box-sizing:border-box; border:1px solid #808080; margin:2px; max-height:40px;vertical-align:middle; max-width:calc(100% - 4px)\"></a>"
		}
		copyrighttext += "</p>"
	})
	if ((copyrighttext != "") && ("true" != "false")) {
		$("#content").append("<p><br></p><hr><div class=\"text\"><p><br></p><p><b>Bildnachweis</b>" + copyrighttext + "</div>")
	}

        if ($("a.kit_fancybox").fancybox)
            $("a.kit_fancybox").fancybox({
        })
        
        if (navigator.userAgent.match(/iPhone|Android/i)) {
            $('.whatsapp_share').css({'display':'inline'})
        }
        icon_top = 5
        icon_opacity = 0.7
        // Feedbackform
        if ($('meta[name=feedback]').attr("content") == "enabled") {
            feedback_text_feedback = 'Feedback'
            feedback_text_hint = 'Ihr Feedback zu dieser Seite ist uns wichtig'
            feedback_text_email = 'Ihre E-Mail: '
            feedback_text_message = 'Ihre Nachricht: '
            feedback_text_send = 'Abschicken&nbsp;&raquo;'
            feedback_text_pull = 'Feedback verfassen'
            feedback_text_sent = 'Feedback verschickt'
            feedback_text_error_sending = 'Fehler beim Versenden:'
            feedback_text_error = 'Fehler: '
            feedback_text_error_mail = 'Keine korrekte Mailadresse!'
            feedback_text_error_captcha = 'Captcha Fehler'
            feedback_text_error_message = 'Keine Nachricht?'
            if ('DEU' != 'DEU') {
                feedback_text_feedback = 'Feedback'
                feedback_text_hint = 'Any feedback from your side will be most welcome'
                feedback_text_email = 'Your mail: '
                feedback_text_message = 'Your message: '
                feedback_text_send = 'Send&nbsp;&raquo;'
                feedback_text_pull = 'write feedback'
                feedback_text_sent = 'Feedback sent'
                feedback_text_error_sending = 'Error sending:'
                feedback_text_error = 'Error: '
                feedback_text_error_mail = 'Incorrect mailaddress!'
                feedback_text_error_captcha = 'Captcha Error'
                feedback_text_error_message = 'No message?'
            }
            var captcha_now
            var captcha_sec
            var captcha_min
            var captcha_zufallszahl_1
            var captcha_zufallszahl_2
            var captcha_loesung
            
            $("body").append('<div id="feedback" style="display:none"><form><div id="feedback_closer"></div><h2>' + feedback_text_feedback + '</h2><p style="text-align:center">' + feedback_text_hint + '</p><label for="feedback_sender">' + feedback_text_email + '</label><input id="feedback_sender" type="text"/><br style="clear:left"><label for="feedback_msg">' + feedback_text_message + '</label><textarea id="feedback_msg"></textarea><br style="clear:left"><label id="feedback_captcha_label" for="feedback_captcha"></label><input id="feedback_captcha" type="text"><br style="clear:left"><input id="feedback_btn" type="button" value="' + feedback_text_send + '" class="btn"/></form><a href="#" id="pull_feedback" class="social-font-icons-randleiste" style="top: 5px;" title="' + feedback_text_pull + '"><i class="fa fa-envelope-o"></i></a></div>')

            $("#feedback").css({'border-radius':'5px', 'border':'1px solid #808080', 'position':'fixed', 'width':'366px', 'top':'24px', 'left':'-364px', 'background-color':'#ffffff', 'z-index':'99'})
            
            //$("#pull_feedback").css({'background':'url("/img/intern/socialmedia.png")  no-repeat 0px 0px transparent', 'display':'block', 'width':'24px', 'height':'24px', 'float':'left', 'position':'absolute', 'top':'5px', 'right':'-24px', 'opacity':icon_opacity})
            //$("#pull_feedback").hover(function(){$(this).css({'opacity':'1'})}, function(){$(this).css({'opacity':icon_opacity})})
            
            $("#feedback_closer").css({'background':'url("/img/intern/socialmedia.png")  no-repeat -24px -72px transparent', 'position':'absolute', 'top':'5px', 'right':'5px', 'width':'17px', 'height':'17px', 'cursor':'pointer', 'opacity':icon_opacity})
            $("#feedback_closer").hover(function(){$(this).css({'opacity':'1'})}, function(){$(this).css({'opacity':icon_opacity})})
            $("#feedback_closer").click(function(){
                $("#feedback").animate({left:"-364px"})
                icon_opacity = 0.7
                
                //$("#pull_feedback").css({'opacity':icon_opacity})
                
            })
            $("#feedback form").css({'float':'left', 'padding':'10px'})
            $("#feedback form label").css({'display':'block', 'float':'left', 'text-align':'right', 'margin-right':'10px', 'width':'80px', 'font-weight':'bold', 'margin-top':'5px'})
            $("#feedback form textarea").css({'font-size':'1.1em', 'width':'250px', 'height':'140px', 'border':'1px solid #009682'})
            $("#feedback form .btn").css({'font-weight':'bold', 'background-image':'linear-gradient(#559d92 0%, #38665f 100%)', 'color':'#ffffff', 'border':'1px solid #009682', 'background-color':'#b3e0da', 'float':'right', 'width':'110px', 'height':'28px', 'border-bottom-right-radius':'5px', 'border-top-left-radius':'5px'})
            $("#feedback form .btn").hover(function(){$(this).css({'background-image':'linear-gradient(#aacec9 0%, #9db5b1 100%)', 'background-color':'#d9efec'})}, function(){$(this).css({'background-image':'linear-gradient(#559d92 0%, #38665f 100%)', 'background-color':'#b3e0da'})})
            $('#feedback form input[type="text"]').css({'width':'250px', 'height':'20px', 'border':'1px solid #009682'})
            $("#feedback h2").css({'text-align':'center'})
            $("#pull_feedback").click(function(){
                $(this).blur()
                if ($("#feedback").css('left') == "0px") {
                    $("#feedback").animate({left:"-364px"})
                }
                else {
                    $("#feedback").animate({left:"0px"})
                    
                    //$("#pull_feedback").animate({"opacity":"1"})
                    
                }
                captcha_now = new Date();
                captcha_sec = captcha_now.getSeconds();
                captcha_min = captcha_now.getMinutes();
                captcha_zufallszahl_1 = captcha_sec % 10;
                captcha_zufallszahl_1 += 1;
                captcha_zufallszahl_2 = (captcha_min + captcha_sec) % 10;
                captcha_zufallszahl_2 += 1;
                captcha_loesung = captcha_zufallszahl_1 + captcha_zufallszahl_2
                $("#feedback_captcha_label").html(captcha_zufallszahl_1 + ' + ' + captcha_zufallszahl_2 + ' = ')
                $("#feedback_captcha").val("")
            })
            $("#feedback_btn").click(function () {
                mailtest = /^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/
                if ((captcha_loesung == $("#feedback_captcha").val()) && (mailtest.test($.trim($("#feedback_sender").val()))) && ($.trim($("#feedback_sender").val()) != "") && ($.trim($("#feedback_msg").val()) != "")) {
                    var request = $.ajax({
                        dataType : 'jsonp', 
                        type: 'POST', 
                        url: "https://www.kit.edu/feedback.php",
                        data: {"url" : $(location).attr('href'),
                            "sender" : $.trim($("#feedback_sender").val()),
                            "feedback" : $.trim($("#feedback_msg").val()),
                            "projguid" : $('meta[name=projguid]').attr("content"),
                            "pageguid" : $('meta[name=pageguid]').attr("content"),
                            "editurl" : $('meta[name=edit]').attr("content")
                        }
                    })
                    request.done(function(data) {
                        if (data["state"] == "success") {
                            alert(feedback_text_sent)
                            $("#feedback_sender").val('')
                            $("#feedback_msg").val('')
                            $("#feedback_captcha").val('')
                            $("#feedback").animate({left:"-364px"});
                        }
                        else
                            alert(feedback_text_error_sending + data.state)
                    })
                    request.fail(function(jqXHR, textStatus) {
                        alert(feedback_text_error + textStatus)
                    })
                }
                else if ((!mailtest.test($.trim($("#feedback_sender").val()))) || ($.trim($("#feedback_sender").val()) == ""))
                    alert(feedback_text_error_mail)
                else if (captcha_loesung != $("#feedback_captcha").val())
                    alert(feedback_text_error_captcha)
                else 
                    alert(feedback_text_error_message)
            })
            $("#feedback").css({'display':'block'})
            //icon_top = 27
            icon_top = 31
            
        }

        if ('true' == 'true') {
            // SocialMedia-Icons an den linken Rand verschieben
            media_icons = new Array()
            counter = 0
            
            //title_rss = 'RSS-Feed'
            //title_youtube = 'YouTube'
            //title_facebook_like = 'Gefällt mir'
            //title_facebook = 'Facebook-Profil'
            //title_twittern = 'URL twittern'
            //title_twitter = 'Twitter-Profil'
            //title_google = 'Google+ Profil'
            //title_google_like = 'Google+ +1'
            
            if ('DEU' != 'DEU') {
            
                //title_rss = 'RSS-Feed'
                //title_youtube = 'YouTube'
                //title_facebook_like = 'I like'
                //title_facebook = 'Facebook-Profile'
                //title_twittern = 'twitter URL'
                //title_twitter = 'Twitter-Profile'
                //title_google = 'Google+ Profile'
                //title_google_like = 'Google+ +1'
                
            }
            // SocialMedia Links einsammeln in Array
            $("#footer-content a.footer-left").each(function() {
              // if ($(this).find("i").attr("class").indexOf("whatsapp") < 0) {
              if (($(this).find("i").attr("class")) && ($(this).find("i").attr("class").indexOf("whatsapp") < 0)) {
              //if ($(this).find("img").attr("src").indexOf("whatsapp") < 0) {
                media_icons[counter] = new Object()
                media_icons[counter]['href'] = $(this).attr("href")
                media_icons[counter]['title'] = $(this).attr("title")
				media_icons[counter]['icon'] = $(this).find("i").attr("class")
				
                //if ($(this).find("img").attr("src").indexOf("rss") > 0) {
                    //media_icons[counter]['type'] = '-24px 0px'
                    //media_icons[counter]['title'] = title_rss
                //}
                
                //if ($(this).find("img").attr("src").indexOf("youtube") > 0) {
                    //media_icons[counter]['type'] = '0px -72px'
                    //media_icons[counter]['title'] = title_youtube
                //}
                
                //if ($(this).find("img").attr("src").indexOf("facebook") > 0) {
                    //if ($(this).find("img").attr("src").indexOf("like") > 0) { // Facebook-Like-Funktion
                        //media_icons[counter]['type'] = '-24px -24px'
                        //media_icons[counter]['title'] = title_facebook_like
                    //}
                    //else { // Facebook-Homepage
                        //media_icons[counter]['type'] = '0px -24px'
                        //media_icons[counter]['title'] = title_facebook
                    //}
                //}
                
                //if ($(this).find("img").attr("src").indexOf("twitter") > 0) {
                    //if ($(this).find("img").attr("src").indexOf("twittern") > 0) { // Seite Twittern
                        //media_icons[counter]['type'] = '-24px -48px'
                        //media_icons[counter]['title'] = title_twittern
                    //}
                    //else { // Twitter-Homepage
                        //media_icons[counter]['type'] = '0px -48px'
                        //media_icons[counter]['title'] = title_twitter
                    //}
                //}
                
                //if ($(this).find("img").attr("src").indexOf("google") > 0) {
                    //if ($(this).find("img").attr("src").indexOf("googleplus") > 0) { // Seite Google+ Profil
                        //media_icons[counter]['type'] = '0px -96px'
                        //media_icons[counter]['title'] = title_google
                    //}
                    //else { // Google+ like
                        //media_icons[counter]['type'] = '-24px -96px'
                        //media_icons[counter]['title'] = title_google_like
                    //}
                //}
                
                counter++
                // $(this).css({'display':'none'}) // SocialMedia-Icons am Seitenende ausblenden
              }
            })
            if (counter > 0) {
                // Falls es noch keinen rechten Rand gibt, diesen erzeugen
                if (!$("#feedback").length) {
                    $("body").append('<div id="feedback"></div>')
                    $("#feedback").css({'height':((counter * 26) + 10) + 'px', 'position':'fixed', 'width':'10px', 'top':'24px', 'left':'-8px', 'background-color':'#ffffff', 'z-index':'99', 'border':'1px solid #808080', 'border-radius':'5px'})
                }
                // An Rand die Icons anlegen
                for (i=0; i<counter; i++) {
                    $("#feedback").append('<a style="top:'+ icon_top + 'px" class="social-font-icons-randleiste" target="medialink" href="' + media_icons[i]["href"] + '" title="' + media_icons[i]['title'] + '"><i class="' + media_icons[i]['icon'] + '"></i></a>');
                    //$("#feedback").append('<a class="media_icon" target="medialink" href="' + media_icons[i]["href"] + '" id="media_icon_' + i + '" title="' + media_icons[i]['title'] + '"></a>');
                    //$("#media_icon_" + i).css({'background':'url("/img/intern/socialmedia.png")  no-repeat ' + media_icons[i]["type"] + ' transparent', 'display':'block', 'width':'24px', 'height':'24px', 'float':'left', 'position':'absolute', 'top':icon_top + 'px', 'right':'-24px', 'opacity':icon_opacity})
                    //$("#media_icon_" + i).hover(function(){$(this).css({'opacity':'1'})}, function(){$(this).css({'opacity':icon_opacity})})
                    //icon_top += 22
                    icon_top += 26
                }
            }
        }
        
        // Tabelle 3 automatisch korrigieren
        $(".tabelle3:not(.dummy)").attr({"cellspacing": "0", "cellpadding": "0"})
        $(".tabelle3:not(.dummy) tr").css("cursor", "pointer")
        $(".tabelle3:not(.dummy) tr").mouseenter( function () { $(this).addClass("hover") });
        $(".tabelle3:not(.dummy) tr").mouseleave( function () { $(this).removeClass("hover") });
        $(".tabelle3:not(.dummy) tr").removeClass("grey");
        $(".tabelle3:not(.dummy) td:has(a)").addClass("link");
        $(".tabelle3:not(.dummy) td:has(a)").removeClass("normal");
        $(".tabelle3:not(.dummy) td:has(a)").removeAttr("onclick");
        $(".tabelle3:not(.dummy)").each( function() {
            $(this).find("tr:not(:has(th)):odd").addClass("grey");
        })
        $(".tabelle3:not(.dummy) td a:has(img)").css({"background-image": "none", "padding": "0"});
        // H4 automatisch korrigieren
        $("h4:not(:has(span.headline-text))").each( function() {
            $(this).html('<span class="headline-text">' + $(this).html() + '</span>')
        })
        
        // Externe Links als solche kennzeichnen
        $("#middle-row a[href*='://']:not(:has(img)):not(:has(svg)):not(.dummy), #right-row a[href*='://']:not(:has(img)):not(:has(svg)):not(.dummy)").each( function() {
            href = $(this).attr("href")
            host = href.split(/\/+/g)[1]
            if (host != location.host) {
                if ((!$(this).attr("target")) && (host.indexOf('.kit.edu') == -1)) $(this).attr("target", "_blank")
                if ($(this).attr("title"))
                    $(this).attr("title", $(this).attr("title") + " (externer Link: " + href + ")");
                else
                    $(this).attr("title", "externer Link: " + href);
                $(this).append('&nbsp;<img class="external_link_symbol" src="/img/intern/icon_external_link.gif" alt="External Link" />')
            }
        })
        $(".external_link_symbol").css({"float": "none", "margin": "0"})
        
        // Floatende Bilder im Text mit Abstand versehen
        $(".text img").each( function() {
            if (($(this).css("float") == 'left') && (parseInt($(this).css("margin-right")) == 0)) {
                $(this).css("margin-right", "6px")
            }
            if (($(this).css("float") == 'right') && (parseInt($(this).css("margin-left")) == 0)) {
                $(this).css("margin-left", "6px")
            }
            if ($(this).attr("longDesc") && ($(this).attr("longDesc") != '')) {
                if (($(this).attr("align")) || ($(this).css("float") != 'none')) {
                    if ($(this).css("float") != 'none') float_side = $(this).css("float")
                    if ($(this).attr("align")) float_side = $(this).attr("align")
                    floater = 'float:' + float_side + ';margin-left: ' + $(this).css("margin-left") + ';margin-right: ' + $(this).css("margin-right")
                    $(this).wrap('<div style="width:' + $(this).attr("width") + 'px;' + floater + '"></div>')
                    $(this).attr("align", "")
                }
                else {
                    $(this).wrap('<span style="padding:6px;display:inline-block;width:' + $(this).attr("width") + 'px"></span>')
                }
                $(this).after('<br><span style="font-size:0.9em">' + $(this).attr("longDesc") + '</span>')
            }
        })
        $(".text img[align=left]").each( function() {
            if (parseInt($(this).css("margin-right")) == 0) {
                $(this).css("margin-right", "6px")
            }
        })
        $(".text img[align=right]").each( function() {
            if (parseInt($(this).css("margin-left")) == 0) {
                $(this).css("margin-left", "6px")
            }
        })
        // Google Analytics ausschaltbar machen
        if ((typeof(_gaq) != "undefined") && (typeof(gmsGAOptState) != "undefined")) {
            $("#footer-content").append('<span class="footer-right" style="border-top-left-radius:5px; border-bottom-right-radius:5px; background-color:#d4defc; margin-top:1px;margin-left: 1em;height:21px"><img id="checkGAActive" style="cursor:pointer; vertical-align:middle" src=""><a href="//www.kit.edu/impressum.php#Google_Analytics" target="GA_Info" title="Information Google Analytics"><img src="//www.kit.edu/img/intern/info.png"  style="vertical-align:middle; margin-left:5px; margin-right:3px"></a></span><script type="text/javascript">gmsInitGASwitch(\'checkGAActive\', \'.kit.edu\')</script>')
            $(".footer-right").css("margin-left", "1em")
        }
        // für Druckausgabe die Infoboxen hinter den Content in unsichtbares DIV kopieren
        // im Print-Stylesheet wird #right-row unsichtbar, dafür #print_infobox sichtbar
        $("#middle-row").append('<div id="print_infobox"></div>')
        $("#print_infobox").css("display", "none")
        $("#print_infobox").append($("div#right-row").html() + '<br style="clear:both">')

        // Responsive Design - Navigation ausblenden
        $("#left-row").prepend('<div id="navigation_hider"><div>Navigation &darr;</div></div>')
        var navigation_toggle = 'initial'
        $("#navigation_hider").click(function() {
            if ($("#menu-box").css('display') == 'none') {
                navigation_toggle = 'open'
                navigation_height = $("#menu-box").css('height');
                $("#menu-box").css({'height':'0px', 'display':'block'})
                $("#menu-box").animate({'height':navigation_height}, 200, function() {
                    $("#menu-box").css({'height':'auto'})
                    $("#navigation_hider div").html('Navigation &uarr;')
                })
            }
            else {
                navigation_toggle = 'closed'
                navigation_height = $("#menu-box").css('height');
                $("#menu-box").css({'height':navigation_height})
                $("#menu-box").animate({'height':'0px'}, 200, function() {
                    $("#menu-box").css({'display':'none', 'height':'auto'})
                    $("#navigation_hider div").html('Navigation &darr;')
                })
            }
        })
        // News auf der Startseite schön anordnen
        function set_news_height() {
            $(".line").css({'display':'none'}).prev('div').css({'clear':'none'})
            $("div").each(function() {
                div_name = $(this).attr('id')+''
                if (div_name.substr(0, 11) == 'homepagebox') {
                    news_height = 0;
                    $(div_name + " .news").each(function() {
                        height = parseInt($(this).css('height'))
                        if (height > news_height) news_height = height
                    })
                    $(div_name + " .news").each(function() {
                        $(this).attr('news_margin', $(this).css('margin-right'))
                    })
                    $(div_name + " .news").css({'min-height':news_height + 5 + 'px', 'margin-right':'12px'})
                }
            })
        }
        function news_float() {
            floats_already = true
            // wait until all images are loaded
            still_wait_for_image = false
            $(".news img").each(function() {
                if (parseInt($(this).css('height')) == 0) still_wait_for_image = true
            })
            if (still_wait_for_image) setTimeout(news_float, 100)
            else set_news_height()
        }
        function news_reset() {
            floats_already = false
            $(".line").css({'display':'block'}).prev('div').css({'clear':'both'})
            $(".news").css({'min-height':'0'})
            $(".news").each(function() {
                $(this).css({'margin-right': $(this).attr('news_margin')})
            })
        }
        var floats_already = false
        if ($("#navigation_hider").css('display') == 'block') {
            news_float()
            $("#menu-box").css({'display':'none'})
            if (!$("#nav-horizontal-2").length) $("#head-image").css({'height':$("#logo").css('height')})
        }
        $(window).resize(function() {
            if (($("#navigation_hider").css('display') == 'none') && ((navigation_toggle == 'closed') || (navigation_toggle == 'initial'))) {
                $("#menu-box").css({'display':'block'})
            }
            if (($("#navigation_hider").css('display') == 'block') && ((navigation_toggle == 'closed') || (navigation_toggle == 'initial'))) {
                $("#menu-box").css({'display':'none'})
            }
            if (($("#navigation_hider").css('display') == 'block') && !floats_already) news_float()
            if (($("#navigation_hider").css('display') == 'block') && (!$("#nav-horizontal-2").length)) $("#head-image").css({'height':$("#logo").css('height')})
            if ($("#navigation_hider").css('display') == 'none') {
                $("#head-image").css({'height':'108px'})
                news_reset()
            }
        })


        if ("true" == "true") {
    		var nav_maxtop = parseInt($("#middle-row").css("height"))
    		var nav_height = parseInt($("#left-row").css("height"))
    		var nav_offset = 120
    		var nav_responsive = ($("#navigation_hider").css("display") == "block")
    		$(window).scroll(function() {
    			if ((!$(".lead_link").length) && (!nav_responsive)) {
    				nav_postop = $(window).scrollTop() - nav_offset
    				if (nav_postop+nav_height > nav_maxtop) nav_postop = nav_maxtop - nav_height
    				if (nav_postop > 0) {
    					$("#left-row").css("margin-top", nav_postop + "px")
    				}
    				else {
    					$("#left-row").css("margin-top", "0px")
    				}
    			}
    			else {
    				$("#left-row").css("margin-top", "0px")
    			}
    		})
    		$(window).resize(function() {
    			nav_responsive = ($("#navigation_hider").css("display") == "block")
    			$(window).trigger("scroll")
    		})
    		$(window).trigger("resize")
		}
    })
}

function changeImg(imgName, imgSrc) {
        document[imgName].src = imgSrc;
        return true;
    }

    function resize_window() {
        document.getElementById('wrapper').style.height = parseInt( document.body.clientHeight) + 'px';
    }

    function noSpam() {
        var a = document.getElementsByTagName("a");
        for (var i = 0; i < a.length; i++) {
            if ( (a[i].href.search(/emailform\b/) != -1) && (a[i].className.search(/force_form\b/) == -1) ) {
                var nodes = a[i].childNodes;
                var email = '';
                for (var j = 0; j < nodes.length; j++) {
                        if (nodes[j].innerHTML) {
                            if (nodes[j].className.search(/caption/) == -1) {
                                email += nodes[j].innerHTML; 
                            }
                        } else {
                            email += nodes[j].data; 
                        }
                }
                email = email.replace(/\u00a0/g, ' '); // &nbsp; in Leerzeichen wandeln
                email = email.replace(/\s/g, '.');
                email = email.replace(/∂/g, '@');
                // a[i].innerHTML = email;
                if (email.search(/@/) != -1) a[i].href = "mailto:" + email;
            }
        }
    }

    function remove_liststyle() {
        if (document.getElementById("right-row")) {
            var lis = document.getElementById("right-row").getElementsByTagName("li");
            for(i=0;i<lis.length;i++) {
                if (lis[i].firstChild.nodeName.toUpperCase() == 'A' ) {
                    lis[i].firstChild.style.backgroundImage = 'none';
                    lis[i].firstChild.style.paddingLeft ='0';
                }
            }
        }
    }
 
    function collapseFAQ() {
/*
        spans = new Array();
        spans = document.getElementsByTagName("p");
        for(i=0; i<spans.length; i++) {
            if (spans[i].id == '') {
                if ((spans[i].className == 'faq_question') || (spans[i].className == 'faq_answer')) {
                    spans[i].id = 'FAQ'; // für IE
                    spans[i].setAttribute('name', 'FAQ'); // für FF
                }
            }
        }
        spans = document.getElementsByTagName("span");
        for(i=0; i<spans.length; i++) {
            if (spans[i].id == '') {
                if ((spans[i].className == 'faq_question') || (spans[i].className == 'faq_answer')) {
                    spans[i].id = 'FAQ'; // für IE
                    spans[i].setAttribute('name', 'FAQ'); // für FF
                }
            }
        }
        spans = document.getElementsByTagName("div");
        for(i=0; i<spans.length; i++) {
            if (spans[i].id == '') {
                if ((spans[i].className == 'faq_question') || (spans[i].className == 'faq_answer')) {
                    spans[i].id = 'FAQ'; // für IE
                    spans[i].setAttribute('name', 'FAQ'); // für FF
                }
            }
        }
        spans = document.getElementsByName("FAQ");
        var counter_question = 0;
        var counter_answer = 0;
        for(i=0; i<spans.length; i++) {
            if (spans[i].className == 'faq_question') {
                spans[i].id = 'faq_question_' + counter_question;
                counter_question++;
                spans[i].onclick = new Function("document.getElementById(this.id + '_answer').style.display = (document.getElementById(this.id + '_answer').style.display == 'none') ? 'block' : 'none';");
                spans[i].style.cursor = 'pointer';
            }
            if (spans[i].className == 'faq_answer') {
                spans[i].id = 'faq_question_' + counter_answer + '_answer';
                counter_answer++;
                spans[i].style.display = 'none';
            }
        }
        (function(){
            var s = window.location.search.substring(1).split('&');
            if(!s.length) return;
            window.$_GET = {};
            for(var i  = 0; i < s.length; i++) {
                var parts = s[i].split('=');
                window.$_GET[unescape(parts[0])] = unescape(parts[1]);
            }
        }())
        counter = 0;
        if ($_GET['faq']) {
            for(i=0; i<spans.length; i++) {
                if (spans[i].className == 'faq_answer') {
                    counter++;
                    if (counter == $_GET['faq']) spans[i].style.display = 'block'
                }
            }
        }
*/
        (function(){
            var s = window.location.search.substring(1).split('&');
            if(!s.length) return;
            window.$_GET = {};
            for(var i  = 0; i < s.length; i++) {
                var parts = s[i].split('=');
                window.$_GET[unescape(parts[0])] = unescape(parts[1]);
            }
        }())
        question_found = false
        question = null
        answer_found = false
        answer = new Array()
        pair = 0
        $(".faq_answer, .faq_question").each( function() {
            if ($(this).hasClass("faq_question")) {
                if (answer_found) { // es wurde vorher schon mindestens eine Antwort gefunden
                    if (question_found) { // zu diesen vorherigen Antworten gab es auch eine Frage
                        pair++ // also haben wir ein Paar aus Frage und Antwort(en) gefunden
                        for (var i=0; i<answer.length; i++) {
                            answer[i].addClass("faq_" + pair + "_" + (i + 1)) // Antworten eindeutig markieren
                            answer[i].removeClass("faq_answer")
                            // if ((!$_GET['faq']) || (pair != $_GET['faq'])) {
                            if ((!$_GET['faq']) || ((pair != $_GET['faq']) && (question.attr("id") != $_GET['faq']))) {
                                answer[i].css({'display':'none'}) // Antworten ausblenden
                            }
                            question.attr("rel", question.attr("rel") + pair + "_" + (i + 1) + " ") // zur Frage gehörige Antworten in REL Attr merken
                        }
                        question.append('&nbsp;<img src="' + question.css("background-image").replace(/url\(/, '').replace(/\)/, '').replace(/\"/g, '') + '"><a id="faq_dummy">&nbsp;</a>')
                        question.css({'color':$("#faq_dummy").css("color"), 'cursor':'pointer'})
                        $("#faq_dummy").remove()
                        question.click( function() {
                            answers = $.trim($(this).attr("rel")).split(" ")
                            for (var i=0; i<answers.length; i++) {
                                if ($(".faq_" + answers[i]).css("display") != "none")
                                    $(".faq_" + answers[i]).css({"display":"none"})
                                else {
                                    $(".faq_" + answers[i]).show()
                                }
                            }
                        })
                        question.removeClass('faq_question')
                    }
                    answer = new Array()
                    answer_found = false
                }
                question_found = true
                question = $(this)
                question.attr("rel", "")
            }
            if ((question_found) && ($(this).hasClass("faq_answer"))) {
                answer_found = true
                answer.push($(this))
            }
        })
        if (answer_found) {
            if (question_found) { // zu diesen vorherigen Antworten gab es auch eine Frage
                pair++ // also haben wir ein Paar aus Frage und Antwort(en) gefunden
                for (var i=0; i<answer.length; i++) {
                    answer[i].addClass("faq_" + pair + "_" + (i + 1)) // Antworten eindeutig markieren
                    answer[i].removeClass("faq_answer")
                    // if ((!$_GET['faq']) || (pair != $_GET['faq'])) {
                    if ((!$_GET['faq']) || ((pair != $_GET['faq']) && (question.attr("id") != $_GET['faq']))) {
                        answer[i].css({'display':'none'}) // Antworten ausblenden
                    }
                    question.attr("rel", question.attr("rel") + pair + "_" + (i + 1) + " ") // zur Frage gehörige Antworten in REL Attr merken
                }
                question.append('&nbsp;<img src="' + question.css("background-image").replace(/url\(/, '').replace(/\)/, '').replace(/\"/g, '') + '"><' + 'a id="faq_dummy">&nbsp;</a>')
                question.css({'color':$("#faq_dummy").css("color"), 'cursor':'pointer'})
                $("#faq_dummy").remove()
                question.click( function() {
                    answers = $.trim($(this).attr("rel")).split(" ")
                    for (var i=0; i<answers.length; i++) {
                        if ($(".faq_" + answers[i]).css("display") != "none")
                            $(".faq_" + answers[i]).css({"display":"none"})
                        else {
                            $(".faq_" + answers[i]).show()
                        }
                    }
                })
                question.removeClass('faq_question')
            } 
            answer = new Array()
            answer_found = false
        }
    }
 