"""
FashionMind — Phase 6A: RAG Fashion Knowledge Base
===================================================
~170 curated fashion-knowledge chunks covering occasion dressing, colour
theory, fit & proportion, fabric & care, body-type styling, capsule
wardrobes, seasonal dressing, accessories, and per-garment guidance.

Retrieval: TF-IDF (1-2 grams) -> Truncated SVD (LSA) -> cosine similarity.
Lightweight on purpose — no GPU, no embedding-model download. At ~170 short
documents this is genuine (if small) retrieval-augmented context, not a
single hard-coded paragraph.

Outputs : models/rag_kb_embeddings.npy
          models/rag_kb_chunks.pkl   ({'chunks': [...], 'tfidf': ..., 'svd': ...})
"""
import os, pickle, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
os.makedirs('models', exist_ok=True)

KNOWLEDGE_BASE = [
    # ── Occasion dressing ────────────────────────────────────────────────────
    "For a beach wedding, choose light breathable fabrics like linen, chiffon, or cotton. Opt for flowy silhouettes in pastel or tropical colours. Avoid heavy fabrics, dark colours, or very formal floor-length gowns that drag on sand.",
    "Business casual means smart trousers or a knee-length skirt with a blouse, shirt, or fine-knit sweater. A blazer is optional but pulls the look together. Avoid jeans, trainers, hoodies, and casual t-shirts.",
    "For a first date, wear something you have worn before and feel confident in — this is not the moment to test an untried outfit. Smart-casual works well: dark jeans with a considered top, or a simple dress with clean shoes.",
    "A job interview calls for professional attire one notch above the company's daily dress code. A well-fitted blazer, tailored trousers or a pencil skirt, and closed-toe shoes project competence. Keep colours neutral and accessories minimal.",
    "Summer festival outfits should prioritise comfort and movement: denim shorts, a flowy dress, or light layers you can tie around your waist. Choose flat sandals or trainers over heels for uneven ground, and bring a packable jacket for the evening.",
    "A cocktail party is semi-formal. A midi dress, a tailored jumpsuit, or dressy separates in rich colours or luxe fabrics such as silk, satin, or velvet all work. Keep hemlines at or below the knee and elevate with heeled shoes.",
    "Black-tie means a floor-length gown or a formal dark suit or tuxedo. Fabrics should have sheen or structure. This is the one occasion where maximal jewellery and formal footwear are expected rather than optional.",
    "For a workout, function comes first: moisture-wicking synthetic or technical fabrics, a supportive sports bra matched to the activity's impact level, and bottoms that allow a full squat without sheerness.",
    "A night out invites bolder choices — a statement top, darker or metallic colours, leather or satin textures, and heeled boots. Pick one focal piece and keep the rest of the outfit simple so it does not compete.",
    "For a funeral or memorial, wear muted, conservative clothing: black, charcoal, navy, or deep grey, with modest coverage and quiet accessories. The goal is to blend in respectfully, not to be noticed.",
    "Smart-casual for the office party sits between your daily workwear and going-out clothes: dark denim or tailored trousers with a silk shirt or fine knit, plus one elevated accessory.",
    "For brunch, lean into relaxed polish — a midi skirt with a tucked tee, wide-leg trousers with a knit, or a casual dress with flat sandals. Daytime events favour lighter colours and less shine.",
    "Travel days call for wrinkle-resistant fabrics (jersey, merino, technical blends), layers you can add or shed through changing temperatures, and slip-on shoes for airport security.",
    "For a garden party, choose midi dresses, linen sets, or tailored shorts in florals or pastels. Wear block heels or wedges rather than stilettos, which sink into grass.",
    "Meeting a partner's family for the first time: aim for neat, modest, and comfortable — a knit with chinos, or a simple dress with a cardigan. Avoid anything with slogans, heavy distressing, or a very revealing cut.",
    "For a client-facing presentation, dress slightly more formally than your audience. Structured shoulders, a defined waist, and closed shoes read as authoritative on camera and in the room.",
    "Weekend errands are the natural home of true casual wear: jeans or joggers, a t-shirt or sweatshirt, and comfortable trainers. Save the effort for occasions that reward it.",
    "For a rooftop or evening summer event, bring a light layer — temperatures drop after sunset. A linen blazer or an oversized knit over a slip dress covers you without heaviness.",
    "Religious services and places of worship usually expect covered shoulders and knees; carry a scarf or a light overshirt you can add at the door.",
    "For a winter formal event, a velvet blazer, a long-sleeved gown, or wool tailoring keeps you warm without a bulky coat over the outfit. A tailored wool overcoat is the right outer layer.",

    # ── Colour theory ───────────────────────────────────────────────────────
    "Analogous colours sit next to each other on the colour wheel and create calm, harmonious outfits — for example navy, cobalt, and teal, or rust, terracotta, and mustard.",
    "Complementary colours sit opposite on the wheel and create high-contrast, energetic combinations: blue and orange, red and green, purple and yellow. Use one as the dominant colour and the other as an accent.",
    "Neutrals — black, white, grey, beige, camel, cream, and navy — go with almost anything and form the backbone of a versatile wardrobe. Build outfits on neutrals, then add colour deliberately.",
    "A monochromatic outfit uses several shades and tints of one colour. Vary the textures — a ribbed knit, a smooth trouser, a suede shoe — so the look has depth rather than flatness.",
    "The 60-30-10 rule balances colour: roughly 60 percent a dominant colour, 30 percent a secondary colour, and 10 percent an accent. It keeps an outfit from looking either bland or chaotic.",
    "Warm skin undertones (veins look greenish, gold jewellery flatters) suit earthy tones, warm reds, coral, olive, and cream. Cool undertones (veins look blue, silver flatters) suit jewel tones, true blue, emerald, and pure white.",
    "Black is the most versatile colour and works from casual to formal, but it can look harsh near a pale face — soften it with a scarf, a lighter collar, or warm makeup.",
    "Head-to-toe white reads fresh and expensive in summer. Keep every white in the outfit the same temperature (all warm ivory, or all cool bright white) so nothing looks dingy by comparison.",
    "Colour blocking means wearing large panels of solid contrasting colour. Limit it to two or three colours, and give each roughly equal visual weight for a deliberate look.",
    "Earth tones — olive, camel, rust, chocolate, ochre — flatter most people and mix easily with denim and cream. They are a low-risk way to wear colour if bright hues feel like too much.",
    "Pastels can wash out fair skin if worn near the face; ground them with a deeper neutral like charcoal or navy, or keep the pastel on the lower half of the body.",
    "Jewel tones — emerald, sapphire, ruby, amethyst — photograph well and suit evening events. They read as richer than primary brights and pair cleanly with black or gold.",
    "When mixing two prints, keep them in the same colour family and vary the scale — one large print, one small — so they read as intentional rather than accidental.",
    "Denim functions as a neutral: mid and dark washes pair with nearly every colour. Very light or heavily faded denim behaves more like a warm off-white.",
    "A pop of an unexpected accent colour — a bag, a shoe, a scarf — lifts an all-neutral outfit without the commitment of a full colourful garment.",
    "Tonal dressing (different depths of related colours, such as oatmeal, camel, and chocolate) looks considered and elongating, and is more forgiving than a single flat colour.",
    "Red draws the eye; wear it where you want attention. A red lip, shoe, or bag is a controlled dose; a full red outfit is a statement that needs confidence and good fit.",
    "Metallics act as neutrals after dark. Gold reads warm and goes with cream, brown, and red; silver reads cool and goes with grey, black, and blue.",

    # ── Fit & proportion ────────────────────────────────────────────────────
    "Fit is the single most important factor in how clothing looks. A well-fitted inexpensive garment almost always looks better than an ill-fitting expensive one.",
    "Balance volume: pair a loose or voluminous top with a slim or fitted bottom, and a fitted top with a wide or full bottom. Loose-on-loose can work but needs a defined waist or an ankle break to avoid looking shapeless.",
    "The tuck — full, half, or French tuck (front only) — into a higher-waisted bottom instantly looks more polished and defines the waist. When in doubt, a loose front tuck flatters most bodies.",
    "High-waisted bottoms lengthen the leg and mark the natural waist. Pair them with tops that hit at or just below the waistband so the high rise is visible.",
    "A shoulder seam should sit at the edge of your shoulder, not down your arm or up your neck. Shoulders are the hardest thing to alter, so prioritise them when buying jackets and coats.",
    "Trouser break — how the hem meets the shoe — sets the formality. No break (hem just above the shoe) looks modern and sharp; a slight break is classic; a full stacked break looks sloppy on tailoring.",
    "Sleeve length on a shirt should end at the wrist bone; on a jacket, show about a centimetre of shirt cuff. Sleeves that swallow the hand make the whole outfit look borrowed.",
    "Cropped tops and jackets work best with high-waisted bottoms so no skin shows at the midriff unless you want it to; the crop should end at or above the narrowest part of your torso.",
    "Vertical elements — a centre pleat, a long cardigan, an open blazer, a monochrome column — draw the eye up and down and make the body look longer and leaner.",
    "Horizontal breaks — a contrasting waistband, a hem at the widest part of the hip or calf, a bold belt — divide the body and draw the eye across. Place them where you want visual width.",
    "When a garment pulls into horizontal creases across a button, zip, or seam, it is too small there. Creases that sag and pool mean it is too big. Aim for a clean, quiet surface.",
    "Tailoring is cheap relative to the improvement it buys. Hemming trousers, taking in a waist, shortening sleeves, and nipping a shirt's side seams transform off-the-rack clothes.",
    "Petite frames look longer in cropped or ankle-length trousers (which show the slimmest part of the leg), shorter jackets, and vertical monochrome. Avoid hems that cut across the mid-calf.",
    "Tall frames can carry maxi lengths, wide legs, longline coats, and horizontal stripes that shorter people cannot. Breaking the body with contrasting halves stops a look from reading as one long line.",
    "For a fuller bust, a V or scoop neckline opens up the chest, and a wrap or a structured (not clingy) fabric skims rather than clings. High crew necks and boxy patch pockets at the bust add volume.",
    "For a smaller bust, boat necks, breast pockets, ruffles, and boxy cuts add the appearance of volume; halters and deep Vs read cleaner.",
    "To define a waist on a straight torso, add a belt, choose pieces with seaming or darts, or tuck a top into a high rise. Wrap dresses and peplum shapes create a waist visually.",
    "For wider hips, A-line skirts, bootcut and wide-leg trousers, and dark bottoms with a lighter or detailed top balance the proportions. Avoid tapered bottoms with side detail at the hip.",
    "An apple or round midsection is flattered by an empire line, a straight shift, an open layer that creates a vertical, and structured fabric that holds its own shape.",
    "Wearing one colour from shoulder to shoe (a 'column of colour') under an open jacket or cardigan is the most reliable elongating trick regardless of body type.",

    # ── Fabric & care ───────────────────────────────────────────────────────
    "Natural fibres — cotton, linen, wool, silk — breathe and age well but wrinkle and can shrink. Synthetics — polyester, nylon, elastane — resist wrinkles and hold shape but trap heat and odour.",
    "Linen is the coolest summer fabric because it wicks moisture and dries fast. It is meant to wrinkle; embrace the rumple or choose a linen-cotton blend for a smoother finish.",
    "Merino wool regulates temperature, resists odour, and can be worn several times between washes. It is ideal for travel and for base layers in cold weather.",
    "Silk is temperature-regulating and drapes beautifully but marks with water and sweat and needs hand-washing or dry cleaning. Wear a base layer under silk in hot weather.",
    "Cashmere is warm for its weight but pills where it rubs (underarms, cuffs, bag straps). De-pill gently with a comb or stone, fold rather than hang, and store with cedar against moths.",
    "Denim lasts longest when washed rarely, inside out, in cold water, and hung to dry. Frequent hot washing and tumble drying fades and weakens the fabric and shrinks the fit.",
    "Structured fabrics (wool suiting, denim, canvas, ponte) hold a shape away from the body and hide lumps. Fluid fabrics (jersey, silk, rayon) reveal every contour — size up or add a layer if that is not the goal.",
    "Check the care label before buying. 'Dry clean only' on something you will wear weekly is a hidden cost; many labels say 'dry clean' to be safe when a cold hand-wash would do.",
    "Wash dark and bright colours cold and inside out to preserve depth. Heat and agitation are what fade dye and break down elastane, so line-drying extends a garment's life significantly.",
    "Wool and knitwear should be dried flat, not hung, because the weight of water stretches the shoulders and hem permanently.",
    "Pilling is caused by friction on loose or short fibres; it is not always a sign of low quality. Turn knits inside out to wash, and keep them away from rough coats and bag straps.",
    "A fabric shaver or a sweater comb restores a pilled knit in minutes and is one of the cheapest ways to make old clothes look new.",
    "Leather needs conditioning a few times a year to stop it drying and cracking, and should dry naturally away from heat if it gets wet. Suede needs a protector spray and a dedicated brush.",
    "Store off-season clothes clean (moths and stains attract pests), in breathable cotton bags rather than plastic, with cedar or lavender rather than mothballs.",
    "Rotate shoes so each pair rests a day between wears; this lets sweat dry and roughly doubles their lifespan. Cedar shoe trees absorb moisture and hold the shape.",
    "Iron or steam clothes inside out when they have prints or dark colours to avoid shine and cracking. Steam is gentler than an iron for delicate and structured pieces.",
    "A cold soak with white vinegar sets dye and removes detergent residue and odour from new or musty garments before the first wear.",

    # ── Capsule wardrobe & building outfits ─────────────────────────────────
    "A capsule wardrobe is a small set of versatile pieces that mix into many outfits. A core women's capsule: white shirt, fine knit, blazer, dark straight jeans, tailored trousers, midi skirt, little black dress, trench, white trainers, ankle boots.",
    "A core men's capsule: white and light-blue oxford shirts, crew and V-neck knits, navy blazer, dark and mid denim, grey and stone chinos, white trainers, brown derbies or chelsea boots, an overcoat.",
    "Buy multiples of the plain pieces you wear out — white tees, black socks, basic knits — and spend the saved decision energy on the few statement pieces that define your style.",
    "Investment pieces earn their price through cost-per-wear: a good coat, well-fitting jeans, leather boots, and a versatile bag are worn constantly for years. Trend pieces should be cheap because their life is short.",
    "The 'one in, one out' rule keeps a wardrobe functional: when you buy something, remove something similar. It forces you to notice what you actually reach for.",
    "If a new purchase does not go with at least three things you already own, it will likely stay unworn. Shop your wardrobe first and buy to fill genuine gaps.",
    "A neutral palette across your wardrobe (pick two or three base colours) means almost everything combines, which is the real secret to getting dressed quickly.",
    "Keep a short mental list of five outfits you know work for your common occasions — work, dinner, casual weekend, formal, travel — so you are never stuck.",
    "Fit and repair what you own before buying more. A tailor, a cobbler, and twenty minutes with a sweater comb often solve the problem you were about to shop for.",
    "The most-worn items in most wardrobes are the plainest: a good t-shirt, well-fitting jeans, a versatile knit, comfortable shoes. Prioritise quality there over novelty elsewhere.",

    # ── Seasonal dressing & layering ───────────────────────────────────────
    "Layering works from thin to thick: a close base layer, an insulating mid layer (knit, fleece), and a weather-resistant outer layer. Each layer should be removable independently as conditions change.",
    "For warmth without bulk, choose thin high-performance layers (merino, down, technical fleece) over one thick jumper. Trapped air between layers is what actually insulates.",
    "In transitional spring and autumn weather, a trench coat, a denim or harrington jacket, or an overshirt bridges cold mornings and mild afternoons.",
    "Summer dressing stays cool with loose cuts, light colours that reflect heat, natural fibres, and less skin coverage only where there is airflow — a loose long sleeve can be cooler than bare arms in direct sun.",
    "Winter accessories do the heavy lifting: a wool scarf, insulated gloves, and a hat prevent most heat loss, letting you wear a lighter coat than you would otherwise need.",
    "Waterproof and water-resistant are different: waterproof keeps rain out entirely (taped seams, high rating); water-resistant handles a light shower. Check before relying on a coat in a downpour.",
    "Transitional footwear — a loafer, an ankle boot, a clean leather trainer — carries an outfit across three seasons and looks more considered than sandals or heavy boots worn out of season.",
    "Tights add a season to dresses and skirts: opaque black or deep colours for autumn and winter, sheer for formal events. Match the tight's weight to the fabric's weight.",
    "A packable down or synthetic jacket that folds into its own pocket is the highest-value travel layer — warm, light, and easy to carry when not needed.",
    "In humid heat, synthetic athletic fabrics can feel worse than cotton or linen because they cling; reserve technical fabrics for active use and choose loose naturals for sitting still.",

    # ── Accessories ────────────────────────────────────────────────────────
    "Accessories are the cheapest way to change an outfit's mood. Add a belt, a scarf, a hat, or a bold earring, then remove one thing before leaving — the last addition is usually one too many.",
    "A belt should either match your shoes or be a deliberate contrast; a near-match that is slightly off looks like a mistake. Belt width should suit the loops and the formality — narrow for tailoring, wider for jeans.",
    "The right bag size signals the occasion: a structured tote for work, a crossbody for daytime and travel, a small clutch or shoulder bag for evening. An oversized slouchy bag undercuts a formal look.",
    "Metals in an outfit should mostly agree — jewellery, belt buckle, watch, bag hardware. Mixing metals can look intentional if repeated at least twice, but one lone silver piece among gold reads as forgotten.",
    "A watch is the one piece of jewellery that works in every setting for everyone; a leather strap dresses up or down more easily than metal.",
    "Scarves add colour near the face (useful with a dark coat) and warmth; a large square can be folded into a neckerchief, a wrap, or tied to a bag handle.",
    "Sunglasses should roughly balance your face shape — angular frames on round faces, rounder frames on angular faces — and sit within the width of your face.",
    "Statement earrings let you skip a necklace; a statement necklace pairs with studs. Wearing both at full volume competes and reads as costume.",
    "Match the formality of your accessories to your clothes: a sports watch and a canvas belt undercut a suit, just as fine jewellery can feel out of place with gym wear.",
    "Hats change proportions: a structured fedora or a beret adds polish, a cap keeps it casual. Whatever the style, it should sit level and fit — a too-big hat ages an outfit instantly.",

    # ── Footwear ──────────────────────────────────────────────────────────
    "Shoes set the formality of an outfit more than any other piece. The same dress reads as three different looks with trainers, ankle boots, or heels.",
    "White leather trainers are the most versatile modern shoe: they work with jeans, chinos, tailored trousers, and midi dresses, and only fail with black tie. Keep them clean — scuffed white trainers drag the whole outfit down.",
    "A pointed toe elongates the leg and looks sharper; a round or almond toe is more comfortable and more casual. Square toes are a trend-driven choice with a shorter shelf life.",
    "Nude or skin-tone heels (matched to your skin, not a generic beige) visually lengthen the leg because there is no colour break at the ankle.",
    "Loafers bridge smart and casual: leather loafers dress up chinos or tailored trousers, and chunky or penny loafers with a lug sole read more relaxed with denim or cropped trousers.",
    "Ankle boots work best when the trouser hem meets the top of the boot or shows a sliver of ankle; a gap of bare shin between a short trouser and a low boot shortens the leg.",
    "Match belt and shoe leather in formal outfits — both black, or both brown in the same rough tone. In casual dress this rule relaxes.",
    "Heel height is a comfort-versus-formality trade-off. A block heel or a low kitten heel gives height and a dressed-up line with far more stability than a stiletto.",
    "Break in new leather shoes at home with thick socks before wearing them out, and carry blister plasters the first few times. Stretching spray and a cobbler's stretch can buy a half size.",
    "Sandals for anything beyond the beach should have some structure — a leather footbed, a defined strap, a slight heel. Flat rubber flip-flops read as strictly casual.",

    # ── Denim ─────────────────────────────────────────────────────────────
    "Dark, uniform-wash denim reads as smart-casual and can pass in many offices; light, faded, or distressed denim is firmly casual.",
    "Straight and slim-straight legs are the most universally flattering denim cut. Skinny jeans emphasise the leg line; wide and barrel legs are fashion-forward and pair best with a fitted or tucked top.",
    "Raw or rigid denim moulds to your body over months and fades where you bend; stretch denim is comfortable immediately but bags out at the knee and seat over a day.",
    "The rise changes everything: a high rise lengthens the leg and holds the stomach, a mid rise suits most people, a low rise is a trend choice that dates quickly.",
    "Hem jeans to your most-worn shoe. A clean straight hem just above the sneaker, or a deliberate crop at the ankle bone, both look intentional; a frayed puddle at the heel does not.",
    "A denim jacket works as a light layer over dresses, knits, and tees. Size it slightly close through the body; an oversized denim jacket is a distinct, more casual statement.",
    "Avoid double denim unless the two washes clearly contrast (a dark jacket over light jeans, or vice versa) — matching washes top and bottom look like a jumpsuit gone wrong.",

    # ── Tops & shirts ─────────────────────────────────────────────────────
    "A crisp white cotton shirt is the hardest-working piece in a wardrobe: under a blazer for work, half-tucked with jeans on the weekend, or open over a tee. Buy two so one is always fresh.",
    "A plain white or ecru t-shirt in a mid-weight cotton, fitted through the shoulder with a hem that hits mid-fly, layers under everything and stands alone. Replace it the moment it goes grey or thin.",
    "Blouses in silk, satin, or crepe add polish to trousers and jeans without a jacket. A slight blouson or a soft bow neckline dresses up faster than a stiff poplin.",
    "A fine-gauge knit (merino, cotton) layers under a jacket where a shirt would bunch and works on its own for smart-casual. A chunky knit is a casual, cosy piece for slim bottoms.",
    "Breton stripes, a chambray shirt, and a grey marl sweatshirt are casual staples that read as 'considered casual' rather than 'gave up'.",
    "Match collar to face and neckline to occasion: a spread collar suits a fuller face and a tie, a button-down is sportier, a grandad collar is relaxed. Deep necklines are for evening; high necks for daytime polish.",
    "A top should skim the body, not cling or tent. If it clings, size up or add a slip; if it tents, look for one with darts, a defined shoulder, or a tuck-friendly hem.",

    # ── Trousers & skirts ────────────────────────────────────────────────
    "Straight-leg trousers are the most versatile cut — they balance most tops and suit most body types. Wide-leg trousers are fashion-forward and need a fitted or tucked top and enough length to nearly graze the floor in heels.",
    "Tailored trousers in wool or a wool blend dress up a knit or tee instantly and are the backbone of smart-casual. A centre crease adds formality; a flat front reads more relaxed.",
    "Chinos sit between jeans and tailored trousers. Stone, navy, and olive are the most useful colours; a slim-straight cut with a clean ankle hem looks current.",
    "A midi skirt (hem between knee and ankle) is the most sophisticated skirt length and works for the office in a pencil or A-line cut and for weekends in a bias-cut slip. Mid-calf hems can shorten the leg — pair with a heel or a pointed flat.",
    "A-line skirts skim over the hips and suit most bodies; pencil skirts define the waist and hip and need the right fit through the seat; pleated skirts add movement and hide a lot.",
    "Wide-leg and palazzo trousers can replace a skirt for formal events — they read as dressy in a fluid fabric and are more comfortable to sit and move in.",
    "Shorts for anywhere other than exercise or the beach should hit within a few centimetres of the knee and be cut in a structured fabric — linen, cotton twill, tailored — rather than jersey.",

    # ── Dresses ──────────────────────────────────────────────────────────
    "A wrap dress is the most universally flattering style: it defines the waist, the neckline adjusts to your comfort, and it suits almost every body shape. A faux-wrap gives the look without the gape.",
    "A shift dress is easy and forgiving but can read boxy; belt it or choose one with subtle shaping if you want a defined waist.",
    "A shirt dress is the daytime workhorse — sleeves rolled and belted for the office, unbuttoned over a swimsuit on holiday. Choose one that fits the shoulders; the rest can be styled.",
    "A slip dress works day and night: layer a tee or fine knit under it and add trainers for day, wear it alone with heels and a jacket for evening.",
    "A little black dress earns its place by being simple enough to restyle: a shape that fits well, a hem that suits your leg, and no fussy details that lock it to one occasion.",
    "Bodycon dresses show everything and are an evening choice; a ponte or scuba fabric holds you in, while a thin jersey needs shapewear and confidence.",
    "For weddings as a guest, avoid white, ivory, and champagne; check if the couple has a colour they have asked guests to avoid; and match the formality on the invitation — a garden ceremony and a ballroom reception call for different dresses.",

    # ── Outerwear ───────────────────────────────────────────────────────
    "A trench coat is the ideal spring and autumn layer — light, water-resistant, and smart enough over tailoring and casual enough over jeans. Classic khaki goes with everything; belt it at the back for a cleaner line.",
    "A wool overcoat (single-breasted, knee-length, in camel, navy, grey, or black) is the one formal outer layer that works over a suit and over a jumper and jeans. Fit the shoulders; have the sleeves altered if needed.",
    "A well-cut blazer is the fastest way to look pulled together — over a tee and jeans, over a dress, or as half of a suit. Navy is the most versatile; a soft unstructured shoulder is more forgiving than a sharp one.",
    "A padded or puffer jacket is the warmest option for its weight; a longline version covers more, and a matte fabric in a dark neutral reads less sporty than a shiny one.",
    "A leather or faux-leather jacket (biker or bomber) adds edge over knits and dresses. Keep it fitted; an oversized leather jacket is a bulky, dominating piece.",
    "A denim or harrington jacket is the lightest true casual layer for mild weather, and layers under a coat for extra warmth in winter.",
    "Match the coat's length and weight to what is under it: a cropped jacket over a long dress needs the dress to carry the look; a long coat over shorts needs the coat mostly closed.",

    # ── Prints, patterns, texture ───────────────────────────────────────
    "Small, dense prints read as texture from a distance and are easier to wear as a main piece; large, high-contrast prints are statements and are often better as an accent or a single garment.",
    "Stripes are near-neutral: a fine navy-and-white stripe combines with other patterns and solids alike. Bold or wide stripes behave like a bold colour and need more space around them.",
    "Vertical stripes lengthen; horizontal stripes widen but also add a relaxed, nautical ease — the effect is smaller than the myth suggests, especially in fine stripes.",
    "Animal print (leopard, snake) works as a neutral in small doses — a shoe, a bag, a belt — pairing with black, denim, camel, and red.",
    "When mixing patterns, anchor them with a shared colour and contrast the scale; a pinstripe with a floral works if both contain the same navy.",
    "Texture is pattern you can wear head-to-toe without clashing: a cable knit, a corduroy trouser, a suede boot in one tonal outfit reads rich, not busy.",
    "Sequins and heavy embellishment are evening-only; balance one sparkly piece with plain, matte everything else.",

    # ── Styling techniques & finishing ─────────────────────────────────
    "Define the waist somewhere in every outfit — a tuck, a belt, a seamed jacket, a high rise — and most looks improve, because an undefined middle reads as shapeless.",
    "The 'third piece' rule: an outfit of just a top and a bottom often looks unfinished; adding a third element — a jacket, a scarf, a layer, a statement shoe — makes it look deliberate.",
    "Cuffing sleeves to just below the elbow and rolling a trouser or jean hem once or twice shows a wrist and an ankle, which lightens and sharpens a look.",
    "Contrast one hard element with one soft: a leather jacket over a floaty dress, or tailored trousers with a relaxed knit. All-hard looks costume-y; all-soft looks like loungewear.",
    "Fit the clothes to your actual current body, not an aspirational size. Clothes that fit now always look better than clothes you are waiting to fit into.",
    "Iron or steam what you wear. A perfectly chosen outfit in a wrinkled fabric looks careless; five minutes with a steamer is the highest-return grooming step.",
    "Keep jewellery and hardware consistent in tone across an outfit, and keep the number of focal points to one or two — a bold shoe and a bold bag and a bold coat all at once cancel each other out.",
    "Dress for the temperature you will be in, not the weather app's headline — offices, restaurants, and transit are often over-air-conditioned or overheated. A removable layer solves this.",
    "When unsure, simplify: fewer colours, cleaner lines, one accessory. A quiet, well-fitted outfit almost never looks wrong; an overworked one often does.",
    "Photograph outfits you like and keep a folder. Over time the pattern of what you actually wear and feel good in becomes obvious, and shopping gets easier.",

    # ── Menswear specifics ────────────────────────────────────────────
    "A suit jacket's bottom button is always left undone. Fasten the top button (of a two-button jacket) when standing and unfasten it when sitting.",
    "A tie should end at the middle of the belt buckle. Tie width should roughly match lapel width — a skinny tie with wide lapels, or vice versa, looks off.",
    "Show a small amount of shirt cuff — about a centimetre — beyond the jacket sleeve. If no cuff shows, the jacket sleeves are too long.",
    "Socks should match the trousers, not the shoes, for a longer leg line in formal wear; bold or patterned socks are a casual, personality choice.",
    "For men, the safest smart-casual formula is a knit or oxford shirt, dark denim or chinos, and a clean leather trainer, loafer, or chelsea boot — with a blazer or overshirt as the optional third piece.",
    "A white and a light-blue shirt, a navy and a grey knit, and one navy blazer cover most of a man's non-casual needs; add colour and pattern only once those basics fit well.",
    "Trouser fit for men: enough room to pinch about two centimetres of fabric at the thigh, a clean line down the front, and a hem with little to no break for a modern look.",

    # ── Body-neutral fit guidance ─────────────────────────────────────
    "There is no universally 'slimming' or 'correct' outfit — dressing well means choosing clothes that fit your current measurements and that you can move and sit comfortably in.",
    "If you are between sizes, buy the size that fits the largest part (shoulders for jackets, hips for trousers, bust for dresses) and have the rest taken in.",
    "Comfort is part of looking good: if you are tugging at a hem, holding in your stomach, or unable to raise your arms, it will show in your posture and how you carry the outfit.",
    "Underwear that fits and disappears under clothes matters more than the clothes themselves — visible lines, a poorly fitted bra, or a waistband that digs will undermine any outfit.",
    "Try clothes on sitting down as well as standing. Trousers, skirts, and fitted dresses that look fine standing can gape, ride up, or strain when you sit.",

    # ── Sustainability & value ───────────────────────────────────────
    "Cost per wear (price divided by how many times you will realistically wear it) is a better guide than price alone. A 200-unit coat worn 200 times is cheaper than a 20-unit top worn twice.",
    "Care extends life more than quality does: washing less and cooler, air-drying, storing properly, and repairing promptly will keep mid-range clothes going for years.",
    "Buy for the wardrobe you have and the life you actually live, not the one you imagine. The occasion-specific piece you 'might need' usually stays unworn while the plain reliable pieces wear out.",
    "Natural single-fibre fabrics (100 percent cotton, wool, linen) are easier to repair, resell, and eventually recycle than blends, which are hard to separate.",
    "A relationship with a good tailor and a cobbler is the most sustainable and economical wardrobe upgrade — hems, zips, resoles, and take-ins cost a fraction of replacement.",

    # ── H&M / mass-retail product knowledge ─────────────────────────────
    "In a large fashion catalogue, garments are grouped by product type (dress, trousers, sweater), by product group (garment upper body, garment lower body, full body), and by garment group (jersey, knitwear, trousers, outerwear) — useful axes for finding alternatives to an item you like.",
    "'Garment Upper body' covers tops, shirts, blouses, knitwear, and jackets; 'Garment Lower body' covers trousers, jeans, shorts, and skirts; 'Garment Full body' covers dresses, jumpsuits, and playsuits.",
    "Colour is described at two levels in retail data: a specific colour group (for example 'Dark Blue', 'Light Pink') and a broader perceived colour master ('Blue', 'Pink') — search the master level for looser matches.",
    "Graphical appearance in product data ('Solid', 'Stripe', 'All over pattern', 'Melange', 'Denim') is a quick filter for how busy a garment is: 'Solid' pieces mix most easily, 'All over pattern' pieces are statements.",
    "For versatile everyday shopping in a big catalogue, filter to solid colours, jersey or knitwear garment groups, and neutral colour masters — that subset gives the highest number of combinable outfits.",
    "Basics (plain tees, simple knits, straight jeans, plain jersey dresses) turn over fast and are cheap to replace; spend a little more on outerwear, shoes, and tailoring, which are worn hardest and hardest to fake.",
    "When an item is out of stock, the closest substitute usually shares the same product type and garment group and a colour in the same perceived colour master, even if the exact colour group differs.",
]


def build():
    print("Building RAG knowledge base...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9, sublinear_tf=True)
    X = tfidf.fit_transform(KNOWLEDGE_BASE)
    # LSA dimensionality: as many as the data supports, capped at 128.
    n_c = min(128, X.shape[0] - 1, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_c, random_state=42)
    X_emb = normalize(svd.fit_transform(X), norm='l2').astype(np.float32)
    print(f"  {len(KNOWLEDGE_BASE)} chunks | tf-idf {X.shape} | LSA {X_emb.shape} | "
          f"variance {svd.explained_variance_ratio_.sum():.1%}")
    np.save('models/rag_kb_embeddings.npy', X_emb)
    pickle.dump({'chunks': KNOWLEDGE_BASE, 'tfidf': tfidf, 'svd': svd},
                open('models/rag_kb_chunks.pkl', 'wb'))
    print("  Saved models/rag_kb_chunks.pkl + models/rag_kb_embeddings.npy ✓")

    def retrieve(q, k=2):
        qv = tfidf.transform([q])
        qe = normalize(svd.transform(qv), norm='l2').astype(np.float32)
        sim = cosine_similarity(qe, X_emb)[0]
        top = sim.argsort()[::-1][:k]
        return [(KNOWLEDGE_BASE[i][:90] + '…', float(sim[i])) for i in top]

    print("\n  Retrieval test:")
    for q in ["what to wear to a beach wedding", "how to mix two patterns",
              "jeans that suit most body types", "keep cashmere from pilling",
              "smart casual for the office", "substitute for an out of stock dress"]:
        r = retrieve(q, k=1)
        print(f"  '{q}'\n     → [{r[0][1]:.2f}] {r[0][0]}")

    print("\nRAG knowledge base complete ✓")


if __name__ == "__main__":
    build()
